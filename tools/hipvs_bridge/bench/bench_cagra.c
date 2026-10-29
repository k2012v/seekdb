/* Standalone benchmark for the seekdb <-> hipVS bridge.
 *
 * It exercises exactly the C ABI that seekdb uses (seekdb_cuvs_build/search/
 * free) without needing a seekdb build or a running observer. Datasets are raw
 * little-endian files:
 *   base     float32[n * dim]      row-major
 *   queries  float32[nq * dim]     row-major
 *   gt       int32[nq * gt_k]      exact L2 ground truth, ascending distance
 *
 * Typical use on the contest image:
 *   bench/build.sh /work/bench/bench_cagra
 *   SEEKDB_CUVS_GRAPH_DEGREE=128 ... /work/bench/bench_cagra \
 *     --bridge /work/bridge/libseekdb_cuvs_bridge.so \
 *     --base base.f32 --n 250000 --dim 256 \
 *     --queries queries.f32 --nq 4096 --gt gt100.i32 --gt-k 100 \
 *     --tag candidate --json-out results.jsonl
 *
 * Build parameters come from the bridge defaults (dimension-aware) unless the
 * SEEKDB_CUVS_* environment variables are set; this lets one binary reproduce
 * every row of the sweep. */
#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void *(*cuvs_build_fn)(const float *, long, long);
typedef int (*cuvs_search_fn)(void *, const float *, long, long,
                              unsigned int *, float *);
typedef void (*cuvs_free_fn)(void *);

static double now_sec(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *read_file(const char *path, size_t bytes)
{
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    fprintf(stderr, "open %s: %s\n", path, strerror(errno));
    return NULL;
  }
  void *buf = malloc(bytes == 0 ? 1 : bytes);
  if (buf == NULL) { fclose(f); return NULL; }
  if (fread(buf, 1, bytes, f) != bytes) {
    fprintf(stderr, "read %s: short read (expected %zu bytes)\n", path, bytes);
    free(buf); fclose(f); return NULL;
  }
  fclose(f);
  return buf;
}

static int parse_long(const char *text, long *out)
{
  char *end = NULL;
  errno = 0;
  long v = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') { return -1; }
  *out = v;
  return 0;
}

static size_t count_hits(const unsigned int *ids, const int *gt,
                         long nq, long k, long gt_k)
{
  size_t hits = 0;
  for (long q = 0; q < nq; ++q) {
    for (long i = 0; i < k; ++i) {
      const int id = (int)ids[q * k + i];
      // Ground truth rows are sorted by distance, so the true top-k are the
      // first k columns; scanning further would inflate Recall@10 into
      // "top-10 hit anywhere in the true top-100".
      const long cols = k < gt_k ? k : gt_k;
      for (long j = 0; j < cols; ++j) {
        if (gt[q * gt_k + j] == id) { ++hits; break; }
      }
    }
  }
  return hits;
}

int main(int argc, char **argv)
{
  const char *base_path = NULL, *query_path = NULL, *gt_path = NULL;
  const char *bridge_path = NULL;
  const char *tag = "", *json_out = NULL;
  long n = 0, dim = 0, nq = 0, gt_k = 100, latency_queries = 64;
  long batch_chunk = 1024;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (strcmp(a, "--base") == 0 && i + 1 < argc) { base_path = argv[++i]; }
    else if (strcmp(a, "--bridge") == 0 && i + 1 < argc) { bridge_path = argv[++i]; }
    else if (strcmp(a, "--n") == 0 && i + 1 < argc) { if (parse_long(argv[++i], &n) != 0) return 2; }
    else if (strcmp(a, "--dim") == 0 && i + 1 < argc) { if (parse_long(argv[++i], &dim) != 0) return 2; }
    else if (strcmp(a, "--queries") == 0 && i + 1 < argc) { query_path = argv[++i]; }
    else if (strcmp(a, "--nq") == 0 && i + 1 < argc) { if (parse_long(argv[++i], &nq) != 0) return 2; }
    else if (strcmp(a, "--gt") == 0 && i + 1 < argc) { gt_path = argv[++i]; }
    else if (strcmp(a, "--gt-k") == 0 && i + 1 < argc) { if (parse_long(argv[++i], &gt_k) != 0) return 2; }
    else if (strcmp(a, "--latency-queries") == 0 && i + 1 < argc) { if (parse_long(argv[++i], &latency_queries) != 0) return 2; }
    else if (strcmp(a, "--batch-chunk") == 0 && i + 1 < argc) { if (parse_long(argv[++i], &batch_chunk) != 0) return 2; }
    else if (strcmp(a, "--tag") == 0 && i + 1 < argc) { tag = argv[++i]; }
    else if (strcmp(a, "--json-out") == 0 && i + 1 < argc) { json_out = argv[++i]; }
    else { fprintf(stderr, "unknown/incomplete argument: %s\n", a); return 2; }
  }
  if (base_path == NULL || query_path == NULL || gt_path == NULL ||
      bridge_path == NULL ||
      n <= 0 || dim <= 0 || nq <= 0 || gt_k <= 0) {
    fprintf(stderr, "usage: %s --bridge F --base F --n N --dim D --queries F --nq NQ "
                    "--gt F [--gt-k 100] [--tag S] [--json-out F]\n", argv[0]);
    return 2;
  }

  void *dl = dlopen(bridge_path, RTLD_NOW | RTLD_LOCAL);
  if (dl == NULL) {
    fprintf(stderr, "dlopen %s: %s\n", bridge_path, dlerror());
    return 2;
  }
  cuvs_build_fn build = (cuvs_build_fn)dlsym(dl, "seekdb_cuvs_build");
  cuvs_search_fn search = (cuvs_search_fn)dlsym(dl, "seekdb_cuvs_search");
  cuvs_free_fn release = (cuvs_free_fn)dlsym(dl, "seekdb_cuvs_free");
  if (build == NULL || search == NULL || release == NULL) {
    fprintf(stderr, "bridge is missing seekdb_cuvs_{build,search,free}: %s\n",
            dlerror());
    return 2;
  }

  float *base = (float *)read_file(base_path, (size_t)n * (size_t)dim * sizeof(float));
  float *queries = (float *)read_file(query_path, (size_t)nq * (size_t)dim * sizeof(float));
  int *gt = (int *)read_file(gt_path, (size_t)nq * (size_t)gt_k * sizeof(int));
  if (base == NULL || queries == NULL || gt == NULL) { return 3; }

  const double t0 = now_sec();
  void *index = build(base, n, dim);
  const double build_s = now_sec() - t0;
  if (index == NULL) {
    fprintf(stderr, "seekdb_cuvs_build failed\n");
    return 4;
  }

  FILE *out = (json_out != NULL) ? fopen(json_out, "a") : NULL;
  if (json_out != NULL && out == NULL) {
    fprintf(stderr, "open %s: %s\n", json_out, strerror(errno));
    release(index); return 5;
  }

  const long ks[] = {10, 100};
  for (size_t ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ++ki) {
    const long k = ks[ki];
    if (k > gt_k) { continue; }
    unsigned int *ids = (unsigned int *)malloc((size_t)nq * (size_t)k * sizeof(unsigned int));
    if (ids == NULL) { release(index); return 6; }

    const long chunk = batch_chunk > 0 ? batch_chunk : nq;
    const double ts0 = now_sec();
    int rc = 0;
    for (long s = 0; s < nq && rc == 0; s += chunk) {
      const long c = (s + chunk <= nq) ? chunk : (nq - s);
      rc = search(index, queries + (size_t)s * (size_t)dim, c, k,
                  ids + (size_t)s * (size_t)k, NULL);
    }
    const double batch_s = now_sec() - ts0;
    if (rc != 0) {
      fprintf(stderr, "seekdb_cuvs_search(nq=%ld, k=%ld) rc=%d\n", nq, k, rc);
      free(ids); release(index); return 7;
    }
    const size_t hits = count_hits(ids, gt, nq, k, gt_k);
    const double recall = (double)hits / (double)(nq * k);

    const long m = latency_queries < nq ? latency_queries : nq;
    const double tl0 = now_sec();
    for (long q = 0; q < m; ++q) {
      if (search(index, queries + (size_t)q * (size_t)dim, 1, k,
                 ids, NULL) != 0) {
        free(ids); release(index); return 8;
      }
    }
    const double latency_ms = (now_sec() - tl0) * 1000.0 / (double)m;
    const double qps = 1000.0 / latency_ms;

    char line[1024];
    snprintf(line, sizeof(line),
             "{\"tag\":\"%s\",\"n\":%ld,\"dim\":%ld,\"nq\":%ld,\"k\":%ld,"
             "\"build_s\":%.4f,\"recall\":%.6f,\"batch_qps\":%.3f,"
             "\"latency_ms\":%.4f,\"qps\":%.3f}\n",
             tag, n, dim, nq, k, build_s, recall,
             (double)nq / (batch_s > 0 ? batch_s : 1e-9), latency_ms, qps);
    fputs(line, stdout);
    if (out != NULL) { fputs(line, out); }
    free(ids);
  }
  if (out != NULL) { fclose(out); }
  release(index);
  free(base); free(queries); free(gt);
  return 0;
}
