// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT
//
// Brute-force ground truth for the kNN-graph benchmarks.
//
// Computes the exact self-kNN graph of a point set (each point's k nearest
// neighbours among the other points, the point itself excluded) using hipVS's
// brute_force index, and dumps it in the same "MI" text format that
// dump_knng() produces, so it can be fed straight to
//   tools/evalulate_knng.py -g <this file> -G MI
//
// Why we need this: the ANN-benchmark HDF5 files ship a query-to-base ground
// truth, not a base-to-base (self) kNN graph, and /p/lustre1/shovan1 only has
// the *-train.txt point files.  Generating the ground truth ourselves also
// pins the metric: whatever normalisation we apply here is the same
// normalisation the NN-Descent / SOLANET / CAGRA runs apply, so all four are
// scored against the same notion of "nearest".
//
// Usage:
//   run_hipvs_single_apu_bf_groundtruth_float_features \
//       -i <points.txt> -p wsv -f l2|ip [-N] -k 100 -G <out.txt> [-b 4096] [-v]

#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/shm_knng_query/data_reader.hpp>
#include <saltatlas/solanet/detail/apu_nn/matrix.hpp>
#include <saltatlas/solanet/detail/cuvs_nn/common.hpp>
#include <saltatlas/solanet/singleton_time_recorder.hpp>

#include "l2_normalize_points.hpp"
#include "solanet_shm_common.hpp"

struct options {
  std::filesystem::path point_files_path;
  std::string           point_file_format;
  std::string           distance_function{"l2"};
  bool                  l2_normalize{false};
  size_t                k{100};
  size_t                batch{4096};
  double                rmm_pool_size_gb{-1};
  std::filesystem::path output_path{};
  bool                  verbose{false};

  void show() const {
    std::cout << "Option:" << std::endl;
    std::cout << "point_files_path: " << point_files_path << std::endl;
    std::cout << "point_file_format: " << point_file_format << std::endl;
    std::cout << "distance_function: " << distance_function << std::endl;
    std::cout << "l2_normalize: " << l2_normalize << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "batch: " << batch << std::endl;
    std::cout << "rmm_pool_size_gb: " << rmm_pool_size_gb << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
  }
};

static bool parse_options(int argc, char* argv[], options& opt,
                          bool& show_usage_flag) {
  int p;
  while ((p = getopt(argc, argv, "i:p:f:k:b:M:G:Nvh")) != -1) {
    switch (p) {
      case 'i': opt.point_files_path = std::filesystem::path(optarg); break;
      case 'p': opt.point_file_format = optarg; break;
      case 'f': opt.distance_function = optarg; break;
      case 'N': opt.l2_normalize = true; break;
      case 'k': opt.k = std::stoul(optarg); break;
      case 'b': opt.batch = std::stoul(optarg); break;
      case 'M': opt.rmm_pool_size_gb = std::stod(optarg); break;
      case 'G': opt.output_path = optarg; break;
      case 'v': opt.verbose = true; break;
      case 'h': show_usage_flag = true; return true;
      default:  show_usage_flag = true; return false;
    }
  }
  if (opt.point_files_path.empty() || opt.point_file_format.empty() ||
      opt.k == 0 || opt.output_path.empty()) {
    std::cerr << "Error: -i, -p, -k and -G are required." << std::endl;
    return false;
  }
  if (opt.distance_function != "l2" && opt.distance_function != "ip") {
    std::cerr << "Error: Unsupported distance function: "
              << opt.distance_function << std::endl;
    return false;
  }
  return true;
}

static void show_usage(const char* prog_name) {
  std::cout << "Usage: " << prog_name
            << " -i <points> -p <format> -f l2|ip [-N] -k <k> -G <out> "
               "[-b <batch>] [-M <rmm_gb>] [-v]"
            << std::endl;
  std::cout << "  Exact self-kNN (brute force). The query point itself is "
               "excluded from its own neighbour list."
            << std::endl;
  std::cout << "  -N: L2-normalize points first (use with -f ip to get a "
               "cosine ground truth)."
            << std::endl;
}

int main(int argc, char* argv[]) {
  options opt;
  bool    show_help = false;
  if (!parse_options(argc, argv, opt, show_help)) return EXIT_FAILURE;
  if (show_help) {
    show_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  opt.show();
  std::cout << std::endl;

  int dev = 0;
  SALTATLAS_HIP_CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop{};
  SALTATLAS_HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  if (opt.verbose) {
    std::printf("Using device %d: %s (gfx: %s)\n", dev, prop.name,
                prop.gcnArchName);
  }

  double pool_size = (opt.rmm_pool_size_gb > 0)
                         ? opt.rmm_pool_size_gb * (1ULL << 30)
                         : rmm::percent_of_free_device_memory(80);
  using rmm_mem_pool_t =
      rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  auto rmm_pool = std::make_unique<rmm_mem_pool_t>(
      rmm::mr::get_current_device_resource(), pool_size);
  rmm::mr::set_current_device_resource(rmm_pool.get());

  std::cout << "\nLoad points" << std::endl;
  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(opt.point_files_path);
  auto points = saltatlas::load_points<
      id_type, id_type, fe_type, e2i_id_map_type,
      saltatlas::solanet::apu_nn::hip_allocator<fe_type>>(
      point_file_paths, opt.point_file_format);
  const size_t n_points = points.num_points();
  const size_t n_dims   = points.num_dimensions();
  std::cout << "Number of points: " << n_points << std::endl;
  std::cout << "Number of dimensions: " << n_dims << std::endl;

  if (opt.l2_normalize) {
    std::cout << "L2-normalizing points..." << std::endl;
    l2_normalize_points(points.data(), n_points, n_dims);
  }

  if (opt.k + 1 > n_points) {
    std::cerr << "Error: k+1 > number of points." << std::endl;
    return EXIT_FAILURE;
  }

  using namespace saltatlas::solanet::cuvs_nn;
  raft::device_resources dev_res;
  auto stream = raft::resource::get_cuda_stream(dev_res);

  const cuvs::distance::DistanceType dist_func =
      (opt.distance_function == "l2")
          ? cuvs::distance::DistanceType::L2Expanded
          : cuvs::distance::DistanceType::InnerProduct;

  auto d_pstore_view = make_dev_matrix_view(points.data(), n_points, n_dims);
  auto d_dataset     = make_const_matrix_view(d_pstore_view);

  std::cout << "\nBuild brute-force index" << std::endl;
  saltatlas::rec_time().start("BF-build");
  cuvs::neighbors::brute_force::index_params bf_params;
  bf_params.metric = dist_func;
  auto bf_index = cuvs::neighbors::brute_force::build(dev_res, bf_params,
                                                      d_dataset);
  SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
  saltatlas::rec_time().stop();  // BF-build

  // Ask for k+1 so that we can drop the query point itself and still have k.
  const size_t topk    = opt.k + 1;
  const size_t batch   = std::min(opt.batch, n_points);
  auto d_neighbors     = raft::make_device_matrix<int64_t, int64_t>(
      dev_res, static_cast<int64_t>(batch), static_cast<int64_t>(topk));
  auto d_distances = raft::make_device_matrix<float, int64_t>(
      dev_res, static_cast<int64_t>(batch), static_cast<int64_t>(topk));

  std::vector<id_type>   gt_ids(n_points * opt.k);
  std::vector<dist_type> gt_dists(n_points * opt.k);
  std::vector<int64_t>   h_nbr(batch * topk);
  std::vector<float>     h_dst(batch * topk);

  std::cout << "\nBrute-force search (k+1 = " << topk << ", batch = " << batch
            << ")" << std::endl;
  saltatlas::rec_time().start("BF-search");
  cuvs::neighbors::brute_force::search_params bf_search;
  for (size_t off = 0; off < n_points; off += batch) {
    const size_t nq = std::min(batch, n_points - off);
    auto q_view     = raft::make_device_matrix_view<const fe_type, int64_t>(
        points.data() + off * n_dims, static_cast<int64_t>(nq),
        static_cast<int64_t>(n_dims));
    auto n_view = raft::make_device_matrix_view<int64_t, int64_t>(
        d_neighbors.data_handle(), static_cast<int64_t>(nq),
        static_cast<int64_t>(topk));
    auto d_view = raft::make_device_matrix_view<float, int64_t>(
        d_distances.data_handle(), static_cast<int64_t>(nq),
        static_cast<int64_t>(topk));

    cuvs::neighbors::brute_force::search(dev_res, bf_search, bf_index, q_view,
                                         n_view, d_view);
    raft::copy(h_nbr.data(), n_view.data_handle(), nq * topk, stream);
    raft::copy(h_dst.data(), d_view.data_handle(), nq * topk, stream);
    raft::resource::sync_stream(dev_res);

    for (size_t q = 0; q < nq; ++q) {
      const size_t sid  = off + q;
      size_t       kept = 0;
      for (size_t j = 0; j < topk && kept < opt.k; ++j) {
        const int64_t nid = h_nbr[q * topk + j];
        if (nid < 0) continue;
        if (static_cast<size_t>(nid) == sid) continue;  // drop self
        gt_ids[sid * opt.k + kept]   = static_cast<id_type>(nid);
        gt_dists[sid * opt.k + kept] = static_cast<dist_type>(h_dst[q * topk + j]);
        ++kept;
      }
      if (kept < opt.k) {
        throw std::runtime_error("Fewer than k neighbours for point " +
                                 std::to_string(sid));
      }
    }
    if (opt.verbose && ((off / batch) % 32 == 0)) {
      std::cout << "  " << off + nq << " / " << n_points << std::endl;
    }
  }
  saltatlas::rec_time().stop();  // BF-search

  print_time_table();
  saltatlas::rec_time().reset();

  // Same "MI" layout as dump_knng(): per point, an id line prefixed by the
  // source id, then a distance line prefixed by a 0.0 placeholder.
  std::cout << "\nDump ground truth to " << opt.output_path << std::endl;
  const auto ids_view =
      saltatlas::solanet::apu_nn::matrix_view<id_type>(gt_ids.data(), n_points,
                                                       opt.k);
  const auto dists_view = saltatlas::solanet::apu_nn::matrix_view<dist_type>(
      gt_dists.data(), n_points, opt.k);
  dump_knng(ids_view, dists_view, opt.output_path, /*dump_distance=*/true);

  return 0;
}
