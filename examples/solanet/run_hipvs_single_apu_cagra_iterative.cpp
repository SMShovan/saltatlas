// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT
//
// Experiment 2: iterative graph building using CAGRA's search() and optimize()
// on a single MI300A APU, via hipVS.
//
//   params.graph_build_params = cagra::graph_build_params::iterative_search_params();
//
// (hipVS: cpp/include/cuvs/neighbors/cagra.hpp, implemented in
//  cpp/src/neighbors/detail/cagra/cagra_build.cuh :: iterative_build_graph)
//
// How it differs from NN-Descent, which matters for how you read the numbers:
//
//   * It starts from a small subgraph (~graph_degree*64 nodes) and DOUBLES the
//     node count each round, running cagra::search() against the current graph
//     and then optimize() to rebuild it.  There is no rho, no delta and no
//     max_iterations; the round count is fixed by log2(N / initial_size).  The
//     -r/-d/-m flags are therefore accepted and ignored, purely so this binary
//     is a drop-in for the same benchmark driver.
//   * The result is a PRUNED, NAVIGABLE graph of degree `k`, not a k-nearest-
//     neighbour graph, and it carries NO distances.  Scored as a kNN graph
//     against brute-force ground truth it will lose to NN-Descent by
//     construction.  The comparison that means something is build time, and
//     recall/QPS of a subsequent cagra::search().
//   * It requires the whole dataset to fit in GPU memory.
//
// The graph is dumped one line per point ("<sid> n1 n2 ... nk"), i.e. the "NI"
// format of tools/evalulate_knng.py.
//
// Usage (same CLI as run_hipvs_single_apu_nndescent):
//   run_hipvs_single_apu_cagra_iterative_float_features \
//       -i <points.txt> -p wsv -f l2|ip [-N] -k 48 [-K <intermediate>] \
//       [-M <rmm_gb>] [-G <out.txt>] [-v]

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
#include <cuvs/neighbors/cagra.hpp>
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
  size_t                k{0};              // output graph degree
  size_t                intermediate_k{0}; // 0 -> same as k
  double                rho{0.5};          // accepted, ignored (no such knob)
  double                delta{0.0001};     // accepted, ignored
  int                   max_iterations{100};  // accepted, ignored
  double                rmm_pool_size_gb{-1};
  bool                  optimize{false};   // accepted, ignored (always pruned)
  std::filesystem::path output_path{};
  bool                  dump_distance{false};  // ignored: no distances exist
  bool                  verbose{false};

  void show() const {
    std::cout << "Option:" << std::endl;
    std::cout << "point_files_path: " << point_files_path << std::endl;
    std::cout << "point_file_format: " << point_file_format << std::endl;
    std::cout << "distance_function: " << distance_function << std::endl;
    std::cout << "l2_normalize: " << l2_normalize << std::endl;
    std::cout << "k (graph_degree): " << k << std::endl;
    std::cout << "intermediate_graph_degree: "
              << (intermediate_k ? intermediate_k : k) << std::endl;
    std::cout << "rmm_pool_size_gb: " << rmm_pool_size_gb << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "(rho/delta/max_iterations are not used by the iterative "
                 "CAGRA build)"
              << std::endl;
  }
};

static bool parse_options(int argc, char* argv[], options& opt,
                          bool& show_usage_flag) {
  int p;
  while ((p = getopt(argc, argv, "i:p:f:k:K:d:r:m:M:oG:NDvh")) != -1) {
    switch (p) {
      case 'i': opt.point_files_path = std::filesystem::path(optarg); break;
      case 'p': opt.point_file_format = optarg; break;
      case 'f': opt.distance_function = optarg; break;
      case 'N': opt.l2_normalize = true; break;
      case 'k': opt.k = std::stoul(optarg); break;
      case 'K': opt.intermediate_k = std::stoul(optarg); break;
      case 'd': opt.delta = std::stod(optarg); break;
      case 'r': opt.rho = std::stod(optarg); break;
      case 'm': opt.max_iterations = std::stoi(optarg); break;
      case 'M': opt.rmm_pool_size_gb = std::stod(optarg); break;
      case 'o': opt.optimize = true; break;
      case 'G': opt.output_path = optarg; break;
      case 'D': opt.dump_distance = true; break;
      case 'v': opt.verbose = true; break;
      case 'h': show_usage_flag = true; return true;
      default:  show_usage_flag = true; return false;
    }
  }
  if (opt.point_files_path.empty() || opt.point_file_format.empty() ||
      opt.distance_function.empty() || opt.k == 0) {
    std::cerr << "Error: Missing required options." << std::endl;
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
            << " -i <points> -p <format> -f l2|ip -k <graph_degree> [options]"
            << std::endl;
  std::cout << "  -N: L2-normalize points before building" << std::endl;
  std::cout << "  -K <n>: intermediate_graph_degree (default: same as -k)"
            << std::endl;
  std::cout << "  -M <gb>: RMM pool size in GB" << std::endl;
  std::cout << "  -G <path>: dump the graph (NI format: '<sid> n1 .. nk')"
            << std::endl;
  std::cout << "  -r/-d/-m/-o/-D: accepted for CLI compatibility, ignored"
            << std::endl;
  std::cout << "  -v: verbose" << std::endl;
}

// One line per point: "<sid> n1 n2 ... nk"  -> evalulate_knng.py -I NI
static void dump_graph_ni(const std::vector<id_type>& ids, const size_t n_points,
                          const size_t k,
                          const std::filesystem::path& output_path) {
  std::ofstream ofs(output_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open output file: " +
                             output_path.string());
  }
  for (size_t sid = 0; sid < n_points; ++sid) {
    ofs << sid;
    for (size_t i = 0; i < k; ++i) ofs << ' ' << ids[sid * k + i];
    ofs << '\n';
  }
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
  if (opt.verbose) {
    std::cout << "RMM pool size (GB): "
              << pool_size / static_cast<double>(1ULL << 30) << std::endl;
  }
  using rmm_mem_pool_t =
      rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  auto rmm_pool = std::make_unique<rmm_mem_pool_t>(
      rmm::mr::get_current_device_resource(), pool_size);
  rmm::mr::set_current_device_resource(rmm_pool.get());

  std::cout << "\nLoad point" << std::endl;
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
    l2_normalize_points(points.data(), n_points, n_dims);
  }

  using namespace saltatlas::solanet::cuvs_nn;
  raft::device_resources dev_res;
  auto stream = raft::resource::get_cuda_stream(dev_res);

  const cuvs::distance::DistanceType dist_func =
      (opt.distance_function == "l2")
          ? cuvs::distance::DistanceType::L2Expanded
          : cuvs::distance::DistanceType::InnerProduct;

  std::cout << "\nBuild KNNG (CAGRA iterative: search() + optimize())"
            << std::endl;

  cuvs::neighbors::cagra::index_params params;
  params.metric                    = dist_func;
  params.graph_degree              = opt.k;
  params.intermediate_graph_degree = opt.intermediate_k ? opt.intermediate_k
                                                        : opt.k;
  params.graph_build_params =
      cuvs::neighbors::cagra::graph_build_params::iterative_search_params();
  // We only want the graph; do not copy the dataset into the index.
  params.attach_dataset_on_build = false;
  params.guarantee_connectivity  = false;

  auto d_pstore_view = make_dev_matrix_view(points.data(), n_points, n_dims);

  saltatlas::rec_time().start("Build-knng");
  auto index = cuvs::neighbors::cagra::build(
      dev_res, params, make_const_matrix_view(d_pstore_view));
  SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
  saltatlas::rec_time().stop();  // Build-knng

  print_time_table();
  saltatlas::rec_time().reset();

  const auto graph        = index.graph();  // device view, (n_points, graph_degree)
  const size_t graph_degree = static_cast<size_t>(graph.extent(1));
  std::cout << "Graph: " << graph.extent(0) << " x " << graph_degree
            << std::endl;

  if (!opt.output_path.empty()) {
    std::cout << "\nDump graph to " << opt.output_path << std::endl;
    if (opt.dump_distance) {
      std::cout << "Note: the iterative CAGRA build produces no distances; "
                   "-D is ignored."
                << std::endl;
    }
    std::vector<id_type> h_ids(static_cast<size_t>(graph.extent(0)) *
                               graph_degree);
    raft::copy(h_ids.data(), graph.data_handle(), h_ids.size(), stream);
    raft::resource::sync_stream(dev_res);
    dump_graph_ni(h_ids, static_cast<size_t>(graph.extent(0)), graph_degree,
                  opt.output_path);
  }

  return 0;
}
