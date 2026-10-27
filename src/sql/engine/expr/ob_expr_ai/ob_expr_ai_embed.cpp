/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_ENG
#include "ob_expr_ai_embed.h"
#include <algorithm>
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{
ObExprAIEmbed::ObExprAIEmbed(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc, 
                        T_FUN_SYS_AI_EMBED, 
                        N_AI_EMBED, 
                        MORE_THAN_ZERO,
                        NOT_VALID_FOR_GENERATED_COL, 
                        NOT_ROW_DIMENSION) 
{
}

ObExprAIEmbed::~ObExprAIEmbed() 
{
}

int ObExprAIEmbed::calc_result_typeN(ObExprResType &type,
                                     ObExprResType *types_stack,
                                     int64_t param_num,
                                     common::ObExprTypeCtx &type_ctx) const 
{
  UNUSED(type_ctx);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3 || param_num < 2)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    if (ob_is_string_type(types_stack[MODEL_IDX].get_type())) {
      types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
    } else {
      ret = OB_ERR_INVALID_TYPE_FOR_OP;
      LOG_USER_ERROR(OB_ERR_INVALID_TYPE_FOR_OP, "ai_embed model id must be a string");
    }
    if (OB_SUCC(ret) && ob_is_string_type(types_stack[CONTENT_IDX].get_type())) {
      types_stack[CONTENT_IDX].set_calc_type(
          ob_is_text_tc(types_stack[CONTENT_IDX].get_type()) ? ObLongTextType : ObVarcharType);
    } else if (OB_SUCC(ret)) {
      ret = OB_ERR_INVALID_TYPE_FOR_OP;
      LOG_USER_ERROR(OB_ERR_INVALID_TYPE_FOR_OP, "ai_embed content must be a string");
    }
    types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    if (param_num == 3) {
      if (types_stack[DIM_IDX].get_type() == ObNullType) {
      } else if (ob_is_integer_type(types_stack[DIM_IDX].get_type())) {
        types_stack[DIM_IDX].set_calc_type(ObIntType);
        types_stack[DIM_IDX].set_precision(10);
        types_stack[DIM_IDX].set_scale(0);
      } else {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be an integer, not a decimal or float");
      }
    }
    type.set_varchar();
    type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    type.set_collation_level(CS_LEVEL_COERCIBLE);
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res) 
{
  INIT_SUCC(ret);
  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_content = nullptr;
  ObDatum *arg_dim = nullptr;
  if (expr.arg_cnt_ == 3 ? OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content, arg_dim))
                         : OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content))) {
  } else if (arg_model_id->is_null()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id is null");
    res.set_null();
  } else if (arg_content->is_null()) {
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo resolved_endpoint;
    const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
    query::ObIAiEndpointResolver *endpoint_resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    ObString model_id = arg_model_id->get_string();
    ObString content;
    if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_,
                                                          temp_allocator,
                                                          *arg_content,
                                                          expr.args_[CONTENT_IDX]->datum_meta_,
                                                          expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(),
                                                          content))) {
    }
    if (model_id.empty() || content.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
      res.set_null();
    }
    int64_t dim = 0;
    ObJsonInt *dim_json = nullptr;
    ObJsonObject *config = nullptr;
    if (OB_FAIL(ret)) {
    } else if (OB_NOT_NULL(arg_dim) && !arg_dim->is_null()) {
      dim = arg_dim->get_int();
      if (dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
        res.set_null();
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, dim, dim_json))) {
      } else if (OB_FAIL(config->add("dimensions", dim_json))) {
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
    } else if (OB_ISNULL(endpoint_resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                   model_id, temp_allocator, resolved_endpoint))) {
    } else {
      ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
      ObString result;
      if (OB_FAIL(model.call_dense_embedding(content, config, result))) {
      } else if (OB_FAIL(ObAIFuncUtils::set_string_result(expr, ctx, res, result))) {
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed_batch(const ObExpr &expr, ObEvalCtx &ctx,
                                       const ObBitVector &skip, const int64_t batch_size)
{
  INIT_SUCC(ret);
  ObDatumVector res_datums = expr.locate_expr_datumvector(ctx);
  ObDatumVector model_datums;
  ObDatumVector content_datums;
  ObDatumVector dim_datums;
  ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
  lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
  if (OB_FAIL(expr.args_[MODEL_IDX]->eval_batch(ctx, skip, batch_size))
      || OB_FAIL(expr.args_[CONTENT_IDX]->eval_batch(ctx, skip, batch_size))
      || (expr.arg_cnt_ == 3 && OB_FAIL(expr.args_[DIM_IDX]->eval_batch(ctx, skip, batch_size)))) {
  } else {
    model_datums = expr.args_[MODEL_IDX]->locate_expr_datumvector(ctx);
    content_datums = expr.args_[CONTENT_IDX]->locate_expr_datumvector(ctx);
    if (expr.arg_cnt_ == 3) {
      dim_datums = expr.args_[DIM_IDX]->locate_expr_datumvector(ctx);
    }
  }

  for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; i++) {
    ObDatum *model_datum = nullptr;
    ObDatum *content_datum = nullptr;
    ObDatum *dim_datum = nullptr;
    if (skip.at(i) || eval_flags.at(i)) {
    } else if (OB_ISNULL(model_datum = model_datums.at(i))
        || OB_ISNULL(content_datum = content_datums.at(i))
        || (expr.arg_cnt_ == 3 && OB_ISNULL(dim_datum = dim_datums.at(i)))) {
      ret = OB_ERR_UNEXPECTED;
    } else if (model_datum->is_null()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id is null");
    } else if (content_datum->is_null()) {
      res_datums.at(i)->set_null();
      eval_flags.set(i);
    } else if (OB_NOT_NULL(dim_datum) && !dim_datum->is_null() && dim_datum->get_int() <= 0) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
    }
  }

  for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; i++) {
    ObString group_model_id;
    int64_t group_dimension = 0;
    ObArray<int64_t> group_rows;
    ObArray<ObString> group_contents;
    if (skip.at(i) || eval_flags.at(i) || model_datums.at(i)->is_null()
        || content_datums.at(i)->is_null()) {
      continue;
    }
    group_model_id = model_datums.at(i)->get_string();
    if (group_model_id.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id is empty");
    } else if (expr.arg_cnt_ == 3 && !dim_datums.at(i)->is_null()) {
      group_dimension = dim_datums.at(i)->get_int();
    }
    for (int64_t j = i; OB_SUCC(ret) && j < batch_size; j++) {
      ObDatum *candidate_dim = expr.arg_cnt_ == 3 ? dim_datums.at(j) : nullptr;
      int64_t candidate_dimension = 0;
      ObString candidate_content;
      if (skip.at(j) || eval_flags.at(j) || model_datums.at(j)->is_null()
          || content_datums.at(j)->is_null()
          || model_datums.at(j)->get_string() != group_model_id) {
        continue;
      }
      if (OB_NOT_NULL(candidate_dim) && !candidate_dim->is_null()) {
        candidate_dimension = candidate_dim->get_int();
      }
      if (candidate_dimension != group_dimension) {
        continue;
      }
      if (OB_FAIL(ObTextStringHelper::read_real_string_data(
          ctx.exec_ctx_, temp_allocator, *content_datums.at(j),
          expr.args_[CONTENT_IDX]->datum_meta_,
          expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(), candidate_content))) {
      } else if (candidate_content.empty()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, input is empty");
      } else if (OB_FAIL(group_rows.push_back(j))) {
      } else if (OB_FAIL(group_contents.push_back(candidate_content))) {
      }
    }

    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo resolved_endpoint;
    query::ObIAiEndpointResolver *endpoint_resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    ObJsonObject *config = nullptr;
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, group_model_id, info))) {
    } else if (OB_ISNULL(endpoint_resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
        group_model_id, temp_allocator, resolved_endpoint))) {
    } else if (group_dimension > 0) {
      ObJsonInt *dimension_json = nullptr;
      if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(
          temp_allocator, group_dimension, dimension_json))) {
      } else if (OB_FAIL(config->add("dimensions", dimension_json))) {
      }
    }

    if (OB_FAIL(ret)) {
      continue;
    }
    ObAIFuncModel model(temp_allocator, *info, resolved_endpoint);
    for (int64_t offset = 0; OB_SUCC(ret) && offset < group_contents.count();
         offset += MAX_EMBED_BATCH_SIZE) {
      int64_t end = std::min(offset + MAX_EMBED_BATCH_SIZE, group_contents.count());
      ObArray<ObString> batch_contents;
      ObArray<ObString> batch_results;
      for (int64_t j = offset; OB_SUCC(ret) && j < end; j++) {
        if (OB_FAIL(batch_contents.push_back(group_contents.at(j)))) {
        }
      }
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(model.call_dense_embedding_vector_v2(
          batch_contents, config, batch_results))) {
      } else if (batch_results.count() != batch_contents.count()) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        for (int64_t j = 0; OB_SUCC(ret) && j < batch_results.count(); j++) {
          int64_t row_idx = group_rows.at(offset + j);
          ObDatum *result_datum = res_datums.at(row_idx);
          ObTextStringDatumResult text_result(
              expr.datum_meta_.type_, &expr, &ctx, result_datum);
          if (OB_FAIL(text_result.init_with_batch_idx(
              batch_results.at(j).length(), row_idx))) {
          } else if (OB_FAIL(text_result.append(batch_results.at(j)))) {
          } else {
            text_result.set_result();
            eval_flags.set(row_idx);
          }
        }
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::cg_expr(ObExprCGCtx &expr_cg_ctx, 
                           const ObRawExpr &raw_expr,
                           ObExpr &rt_expr) const 
{
  int ret = OB_SUCCESS;
  // TODO: support schema version match in plan cache for ai func
  // const ObRawExpr *model_key = raw_expr.get_param_expr(0);
  // if (OB_NOT_NULL(model_key)
  //     && (model_key->is_static_scalar_const_expr() || model_key->is_const_expr())
  //     && model_key->get_expr_type() != T_OP_GET_USER_VAR &&
  //     OB_NOT_NULL(expr_cg_ctx.schema_guard_)) {
  //   ObIAllocator *allocator = expr_cg_ctx.allocator_;
  //   ObExecContext *exec_ctx = expr_cg_ctx.session_->get_cur_exec_ctx();
  //   ObObj const_data;
  //   bool got_data = false;
  //   ObAIFuncExprInfo *info = nullptr;
  //   if (OB_ISNULL(allocator)) {
  //     ret = OB_ERR_UNEXPECTED;
  //     LOG_WARN("allocator is null", K(ret));
  //   } else if (OB_FAIL(ObSQLUtils::calc_const_or_calculable_expr(exec_ctx,
  //                                                         model_key,
  //                                                         const_data,
  //                                                         got_data,
  //                                                         *allocator))) {
  //     LOG_WARN("failed to calc offset expr", K(ret));
  //   } else if (!got_data || const_data.is_null()) {
  //   } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //     LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //   } else {
  //     rt_expr.extra_info_ = info;
  //   }
  // }

  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIEmbed::eval_ai_embed;
    rt_expr.eval_batch_func_ = ObExprAIEmbed::eval_ai_embed_batch;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
