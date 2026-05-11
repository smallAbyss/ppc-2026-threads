#pragma once
#include "luzan_e_double_sparse_matrix_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luzan_e_double_sparse_matrix_mult {

class LuzanEDoubleSparseMatrixMultALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }
  explicit LuzanEDoubleSparseMatrixMultALL(const InType &in);

  static void BroadcastMatrix(SparseMatrix &m, int root = 0);
  // static SparseMatrix CalcProdALL(const SparseMatrix &a, const SparseMatrix &b);
  static SparseMatrix CalcProdMPIOMP(const SparseMatrix &a_in,
                             const SparseMatrix &b_in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace luzan_e_double_sparse_matrix_mult
