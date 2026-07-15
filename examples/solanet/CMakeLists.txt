if (SALTATLAS_USE_APU)
    add_fe_typed_exe(run_solanet_single_apu_nndescent float)
    setup_hip_target(${target_name})

    add_fe_typed_exe(run_solanet_single_apu_nndescent uint8_t)
    setup_hip_target(${target_name})

    add_fe_typed_exe(run_hipvs_single_apu_nndescent float)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})

    add_fe_typed_exe(run_hipvs_single_apu_nndescent uint8_t)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})

    add_fe_typed_ygm_exe(run_solanet_apu float)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})

    add_fe_typed_ygm_exe(run_solanet_apu uint8_t)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})
    # Experiment 2: iterative CAGRA graph build (search() + optimize())
    add_fe_typed_exe(run_hipvs_single_apu_cagra_iterative float)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})

    # Brute-force ground truth for the kNN-graph benchmarks
    add_fe_typed_exe(run_hipvs_single_apu_bf_groundtruth float)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})

endif()
