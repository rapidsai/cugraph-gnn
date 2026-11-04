from cuda.bindings import driver as cu


def chk(r):
    err = r[0] if isinstance(r, tuple) else r
    return err, (r[1:] if isinstance(r, tuple) else ())


def attr(a, dev):
    return cu.cuDeviceGetAttribute(a, dev)


cu.cuInit(0)
_, dev = cu.cuDeviceGet(0)
_, ctx = cu.cuDevicePrimaryCtxRetain(dev)
cu.cuCtxSetCurrent(ctx)
print("driver", cu.cuDriverGetVersion())
A = cu.CUdevice_attribute
for name in [
    "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
    "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED",
    "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED",
    "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED",
    "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED",
]:
    print(name, attr(getattr(A, name), dev))

H = cu.CUmemAllocationHandleType
for htype in ["CU_MEM_HANDLE_TYPE_NONE", "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR"]:
    for rdma in (0, 1):
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0
        prop.requestedHandleTypes = getattr(H, htype)
        prop.allocFlags.gpuDirectRDMACapable = rdma
        err, (gran,) = chk(
            cu.cuMemGetAllocationGranularity(
                prop,
                cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )
        size = gran
        steps = []
        err, rest = chk(cu.cuMemCreate(size, prop, 0))
        steps.append(("create", err))
        if err == cu.CUresult.CUDA_SUCCESS:
            (h,) = rest
            err, (ptr,) = chk(cu.cuMemAddressReserve(size, gran, 0, 0))
            steps.append(("reserve", err))
            err, _ = chk(cu.cuMemMap(ptr, size, 0, h, 0))
            steps.append(("map", err))
            desc = cu.CUmemAccessDesc()
            desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            desc.location.id = 0
            desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            err, _ = chk(cu.cuMemSetAccess(ptr, size, [desc], 1))
            steps.append(("setaccess", err))
            if htype.endswith("DESCRIPTOR"):
                err, _ = chk(
                    cu.cuMemExportToShareableHandle(h, getattr(H, htype), 0)
                )
                steps.append(("export", err))
            cu.cuMemUnmap(ptr, size)
            cu.cuMemAddressFree(ptr, size)
            cu.cuMemRelease(h)
        print(htype, "rdma", rdma, "gran", gran, [(s, e.name) for s, e in steps])
