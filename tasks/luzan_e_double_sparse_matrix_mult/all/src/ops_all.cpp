#include <cmath>
#include <cstddef>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "luzan_e_double_sparse_matrix_mult/common/include/common.hpp"
#include "luzan_e_double_sparse_matrix_mult/all/include/ops_all.hpp"

#include <iostream>



namespace luzan_e_double_sparse_matrix_mult {

LuzanEDoubleSparseMatrixMultALL::LuzanEDoubleSparseMatrixMultALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  // GetOutput() = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: broadcast a SparseMatrix from `root` to every rank.
// Non-root ranks pass in a default-constructed matrix; it gets filled here.
// ─────────────────────────────────────────────────────────────────────────────
void LuzanEDoubleSparseMatrixMultALL::BroadcastMatrix(SparseMatrix &m, int root) {
  //std::cout << "BroadcastMatrix\n";
  MPI_Bcast(&m.rows, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&m.cols, 1, MPI_INT, root, MPI_COMM_WORLD);

  int nnz     = static_cast<int>(m.value.size());
  int ci_size = static_cast<int>(m.col_index.size());
  MPI_Bcast(&nnz,     1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&ci_size, 1, MPI_INT, root, MPI_COMM_WORLD);

  m.value.resize(nnz);
  m.row.resize(nnz);
  m.col_index.resize(ci_size);

  MPI_Bcast(m.value.data(),     nnz,     MPI_DOUBLE,   root, MPI_COMM_WORLD);
  MPI_Bcast(m.row.data(),       nnz,     MPI_UNSIGNED, root, MPI_COMM_WORLD);
  MPI_Bcast(m.col_index.data(), ci_size, MPI_UNSIGNED, root, MPI_COMM_WORLD);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main routine.
//   • Call on every MPI rank.
//   • a_in / b_in only need to be valid on rank 0; other ranks pass empty ones.
//   • Only rank 0 returns a populated result; other ranks return an empty matrix.
// ─────────────────────────────────────────────────────────────────────────────
SparseMatrix LuzanEDoubleSparseMatrixMultALL::CalcProdMPIOMP(const SparseMatrix &a_in,
                             const SparseMatrix &b_in) {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  //std::cout << "step1\n";

  // ── 1. Broadcast A and B to every rank ──────────────────────────────────
  SparseMatrix a = (rank == 0) ? a_in : SparseMatrix{};
  SparseMatrix b = (rank == 0) ? b_in : SparseMatrix{};
  BroadcastMatrix(a, 0);
  BroadcastMatrix(b, 0);
  //std::cout << "step2\n";

  // ── 2. Distribute B columns across ranks ─────────────────────────────────
  //   counts[i]  – number of columns owned by rank i
  //   displs[i]  – global column index where rank i starts
  int b_cols = b.cols;
  std::vector<int> counts(nprocs), displs(nprocs, 0);
  {
    int base = b_cols / nprocs;
    int rem  = b_cols % nprocs;
    for (int i = 0; i < nprocs; i++)
      counts[i] = base + (i < rem ? 1 : 0);
    for (int i = 1; i < nprocs; i++)
      displs[i] = displs[i - 1] + counts[i - 1];
  }

  int my_col_start = displs[rank];
  int my_col_count = counts[rank];

    //std::cout << "step3\n";
  // ── 3. Local OMP computation (identical kernel to CalcProdOMP) ───────────
  std::vector<std::vector<double>>   values_per_col(my_col_count);
  std::vector<std::vector<unsigned>> rows_per_col  (my_col_count);

 
#pragma omp parallel for schedule(static) default(none) \
    shared(a, b, values_per_col, rows_per_col, my_col_start, my_col_count)
  for (int lc = 0; lc < my_col_count; lc++) {
    int b_col = my_col_start + lc;

    std::vector<double> tmp_col(a.rows, 0.0);

    unsigned b_rs = b.col_index[b_col];
    unsigned b_re = b.col_index[b_col + 1];

    for (unsigned bp = b_rs; bp < b_re; bp++) {
      double   b_val = b.value[bp];
      unsigned b_row = b.row[bp];

      unsigned a_rs = a.col_index[b_row];
      unsigned a_re = a.col_index[b_row + 1];

      for (unsigned ap = a_rs; ap < a_re; ap++)
        tmp_col[a.row[ap]] += a.value[ap] * b_val;
    }

    for (unsigned int i = 0; i < a.rows; i++) {
      if (std::fabs(tmp_col[i]) > kEPS) {
        values_per_col[lc].push_back(tmp_col[i]);
        rows_per_col[lc].push_back(static_cast<unsigned>(i));
      }
    }
  }

    //std::cout << "step4\n";

  // ── 4. Serialize local result into flat arrays ───────────────────────────
  //   col_nnz[lc]  – nnz in local column lc (used to reconstruct col_index)
  //   local_vals / local_rows_flat – all values/row-indices, column by column
  std::vector<int>      col_nnz(my_col_count);
  std::vector<double>   local_vals;
  std::vector<unsigned> local_rows_flat;

  for (int lc = 0; lc < my_col_count; lc++) {
    col_nnz[lc] = static_cast<int>(values_per_col[lc].size());
    local_vals.insert(local_vals.end(),
                      values_per_col[lc].begin(), values_per_col[lc].end());
    local_rows_flat.insert(local_rows_flat.end(),
                           rows_per_col[lc].begin(), rows_per_col[lc].end());
  }
  int local_nnz = static_cast<int>(local_vals.size());

    //std::cout << "step5\n";

  // ── 5. Gather col_nnz from every rank → rank 0 gets b_cols ints in order ─
  //   counts / displs already describe the column distribution, so they also
  //   describe how many ints each rank contributes.
  std::vector<int> global_col_nnz(rank == 0 ? b_cols : 0);

  MPI_Gatherv(col_nnz.data(),       my_col_count, MPI_INT,
              global_col_nnz.data(), counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  // ── 6. Gather flat value / row arrays ────────────────────────────────────
  //   First collect nnz per rank, then do a single Gatherv for values & rows.
    //std::cout << "step6\n";

  std::vector<int> nnz_counts(nprocs, 0), nnz_displs(nprocs, 0);
  MPI_Gather(&local_nnz, 1, MPI_INT,
             nnz_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total_nnz = 0;
  if (rank == 0) {
    for (int i = 1; i < nprocs; i++)
      nnz_displs[i] = nnz_displs[i - 1] + nnz_counts[i - 1];
    total_nnz = nnz_displs[nprocs - 1] + nnz_counts[nprocs - 1];
  }

  std::vector<double>   global_vals(rank == 0 ? total_nnz : 0);
  std::vector<unsigned> global_rows(rank == 0 ? total_nnz : 0);

  MPI_Gatherv(local_vals.data(),  local_nnz, MPI_DOUBLE,
              global_vals.data(), nnz_counts.data(), nnz_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Gatherv(local_rows_flat.data(), local_nnz, MPI_UNSIGNED,
              global_rows.data(),     nnz_counts.data(), nnz_displs.data(),
              MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  //std::cout << "step7\n";

  // ── 7. Rank 0 assembles the final CCS matrix ─────────────────────────────
  SparseMatrix c;
  if (rank == 0) {
    c.rows      = a.rows;
    c.cols      = b.cols;
    c.value     = std::move(global_vals);
    c.row       = std::move(global_rows);

    c.col_index.reserve(b_cols + 1);
    c.col_index.push_back(0);
    for (int j = 0; j < b_cols; j++)
      c.col_index.push_back(c.col_index.back() + global_col_nnz[j]);
  }

  return c;  // empty on non-root ranks
}


bool LuzanEDoubleSparseMatrixMultALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return true;
  }
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  return a.GetCols() == b.GetRows() && a.GetCols() != 0 && a.GetRows() != 0 && b.GetCols() != 0;
}

bool LuzanEDoubleSparseMatrixMultALL::PreProcessingImpl() {
  return true;
}

bool LuzanEDoubleSparseMatrixMultALL::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());

  GetOutput() = CalcProdMPIOMP(a, b);
  return true;
}

bool LuzanEDoubleSparseMatrixMultALL::PostProcessingImpl() {
    // //std::cout << "POSTPROC\n\n";

  return true;
}

}  // namespace luzan_e_double_sparse_matrix_mult
