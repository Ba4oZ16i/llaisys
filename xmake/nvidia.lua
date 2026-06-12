add_requires("cuda")

target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    add_packages("cuda")
    set_languages("cxx17")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC", {force = true})
    end

    -- CUDA source files are compiled in the llaisys shared library target
    -- to trigger automatic device linking (nvcc -dlink)

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_packages("cuda")
    set_languages("cxx17")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC", {force = true})
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()