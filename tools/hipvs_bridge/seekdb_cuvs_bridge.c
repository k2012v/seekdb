/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* Build parameters for the contest workloads (250K x {128,256} FP32, L2).
 *
 * The shape of a CAGRA graph is controlled by three knobs:
 *   graph_degree               - final out-degree (also the query-time fan-out)
 *   intermediate_graph_degree  - degree used while refining the graph
 *   nn_descent_niter           - NN-descent refinement rounds
 *
 * The frozen baseline spends 100 NN-descent iterations refining a 256-degree
 * intermediate graph before pruning it back to 128. Sweeps on the contest GPU
 * (Radeon PRO W7900D, ROCm 7.2.4, hipVS) show two things:
 *   - the intermediate degree carries most of the recall: dropping it to 128
 *     while keeping the final degree at 128 loses ~0.04 recall@10 on 250K x
 *     256 (0.9997 -> 0.9568), so igd stays at 256;
 *   - NN-descent converges long before 100 iterations on both contest shapes:
 *     niter=40 costs ~1.9x less build time than the baseline while keeping
 *     recall at or above it (250K x 128: 0.99998 vs 0.99995; 250K x 256:
 *     0.9998 vs 0.9997).
 * Defaults are therefore 128/256/40 for both shapes; the final degree must stay
 * at 128 because ef_search=64 walks the same graph at query time.
 * See docs/gpu-vector-index-hipvs-cuvs/build-optimization-report.md.
 *
 * The environment overrides exist so the same binary can reproduce the sweep
 * in the report. The evaluation harness does not set them, so the defaults
 * below are what the contest runs. */
static int cuvs_env_int(const char *name, int def_val, int min_val, int max_val)
{
  const char *text = getenv(name);
  if (text == NULL || text[0] == '\0') return def_val;
  char *end = NULL;
  long value = strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < min_val || value > max_val) return def_val;
  return (int)value;
}

static void cuvs_default_build_params(long dim, int *graph_degree,
                                      int *intermediate_graph_degree,
                                      int *nn_descent_niter)
{
  (void)dim;
  *graph_degree = 128;
  *intermediate_graph_degree = 256;
  *nn_descent_niter = 40;
}

static void cuvs_set_build_params(cuvsCagraIndexParams_t ip, long dim)
{
  int graph_degree = 0;
  int intermediate_graph_degree = 0;
  int nn_descent_niter = 0;
  cuvs_default_build_params(dim, &graph_degree, &intermediate_graph_degree,
                            &nn_descent_niter);
  ip->build_algo = NN_DESCENT;
  ip->graph_degree = cuvs_env_int("SEEKDB_CUVS_GRAPH_DEGREE", graph_degree, 16, 256);
  ip->intermediate_graph_degree =
      cuvs_env_int("SEEKDB_CUVS_INTERMEDIATE_GRAPH_DEGREE",
                   intermediate_graph_degree, 32, 512);
  if (ip->intermediate_graph_degree < ip->graph_degree) {
    ip->intermediate_graph_degree = ip->graph_degree;
  }
  ip->nn_descent_niter = cuvs_env_int("SEEKDB_CUVS_NN_DESCENT_NITER",
                                      nn_descent_niter, 1, 1000);
}

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  /* Search scratch reused across calls. Without it every single-query search
   * adds three device alloc/free pairs plus a fresh cuVS search-params object
   * to the critical path of every small top-k query.
   * seekdb serializes searches on a handle, but the bridge also guards this
   * state so the standalone batch_knn/knn entry points stay safe. */
  pthread_mutex_t        mu;
  cuvsCagraSearchParams_t sp;
  void                  *qd;
  void                  *nd;
  void                  *dd;
  size_t                 qd_bytes;
  size_t                 nd_bytes;
  size_t                 dd_bytes;
} sk_cuvs_index;

/* Grow a cached device buffer to at least `need_bytes` without losing the old
 * allocation if the new one fails. Returns 0 on success. */
static int sk_cuvs_reserve(cuvsResources_t res, void **buf, size_t *cap, size_t need_bytes)
{
  void *next = NULL;
  if (*cap >= need_bytes) return 0;
  if (cuvsRMMAlloc(res, &next, need_bytes) != CUVS_SUCCESS) return -1;
  if (*buf != NULL) cuvsRMMFree(res, *buf, *cap);
  *buf = next;
  *cap = need_bytes;
  return 0;
}

static int sk_cuvs_init_search_state(sk_cuvs_index *h)
{
  if (pthread_mutex_init(&h->mu, NULL) != 0) return -1;
  if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) {
    pthread_mutex_destroy(&h->mu);
    return -1;
  }
  /* Search params are identical for every query of an index; create them once.
   * itopk_size only ever grows (topk may differ between statements). */
  h->sp->search_width = (size_t)cuvs_env_int("SEEKDB_CUVS_SEARCH_WIDTH", 4, 1, 32);
  h->sp->itopk_size = (size_t)cuvs_env_int("SEEKDB_CUVS_SEARCH_ITOPK", 512, 64, 8192);
  {
    int max_queries = cuvs_env_int("SEEKDB_CUVS_SEARCH_MAX_QUERIES", 0, 0, 1 << 20);
    h->sp->max_queries = (size_t)max_queries;
  }
  return 0;
}

static void sk_cuvs_destroy_search_state(sk_cuvs_index *h)
{
  if (h->sp != NULL) {
    cuvsCagraSearchParamsDestroy(h->sp);
    h->sp = NULL;
  }
  if (h->qd != NULL) { cuvsRMMFree(h->res, h->qd, h->qd_bytes); h->qd = NULL; h->qd_bytes = 0; }
  if (h->nd != NULL) { cuvsRMMFree(h->res, h->nd, h->nd_bytes); h->nd = NULL; h->nd_bytes = 0; }
  if (h->dd != NULL) { cuvsRMMFree(h->res, h->dd, h->dd_bytes); h->dd = NULL; h->dd_bytes = 0; }
  pthread_mutex_destroy(&h->mu);
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  cuvs_set_build_params(h->ip, dim);
  fprintf(stderr, "[seekdb_cuvs] build n=%ld dim=%ld gd=%d igd=%d niter=%d\n",
          n, dim, (int)h->ip->graph_degree, (int)h->ip->intermediate_graph_degree,
          (int)h->ip->nn_descent_niter);
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  if (sk_cuvs_init_search_state(h) != 0) goto fail_idx;
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !query || !out_ids || nq <= 0 || topk <= 0) return 1;
  int rc = 0;
  const size_t qd_bytes = sizeof(float) * (size_t)nq * (size_t)h->dim;
  const size_t out_bytes = sizeof(unsigned) * (size_t)nq * (size_t)topk;
  const size_t dist_bytes = sizeof(float) * (size_t)nq * (size_t)topk;
  pthread_mutex_lock(&h->mu);
  if (sk_cuvs_reserve(h->res, &h->qd, &h->qd_bytes, qd_bytes) != 0) { rc=10; goto done; }
  if (sk_cuvs_reserve(h->res, &h->nd, &h->nd_bytes, out_bytes) != 0) { rc=11; goto done; }
  /* out_dist is optional, but the distance buffer is always produced by cuVS. */
  if (sk_cuvs_reserve(h->res, &h->dd, &h->dd_bytes, dist_bytes) != 0) { rc=12; goto done; }
  if (cudaMemcpy(h->qd, query, qd_bytes, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=h->qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=h->nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=h->dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    {
      size_t required_itopk = (size_t)topk * 4;
      size_t itopk_floor = (size_t)cuvs_env_int("SEEKDB_CUVS_SEARCH_ITOPK", 512, 64, 8192);
      if (required_itopk < itopk_floor) required_itopk = itopk_floor;
      if (h->sp->itopk_size < required_itopk) h->sp->itopk_size = required_itopk;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, h->nd, out_bytes, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist && cudaMemcpy(out_dist, h->dd, dist_bytes, cudaMemcpyDefault) != cudaSuccess) { rc=17; goto done; }
  }
done:
  pthread_mutex_unlock(&h->mu);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  sk_cuvs_destroy_search_state(h);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  if (sk_cuvs_init_search_state(index) != 0) goto fail_idx;
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
