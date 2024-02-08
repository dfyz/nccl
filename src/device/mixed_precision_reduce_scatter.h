#include "primitives.h"

#include <cassert>

template<typename TInput, typename T>
struct Converter {
  static constexpr size_t Scale = sizeof(T)/sizeof(TInput);

  template<int Size>
  __device__ static BytePack<Size*Scale> upcast(BytePack<Size> pack) {
    if constexpr (Size < sizeof(TInput)) {
      static_assert("Shouldn't happen");
    } else if constexpr (Size == sizeof(TInput)) {
      return toPack<T>(static_cast<T>(fromPack<TInput>(pack)));
    } else {
      BytePack<2*Size> ret;
      ret.half[0] = Converter<TInput, T>::upcast(pack.half[0]);
      ret.half[1] = Converter<TInput, T>::upcast(pack.half[1]);
      return ret;
    }
  }

  template<int Size>
  __device__ static BytePack<Size/Scale> downcast(BytePack<Size> pack) {
    if constexpr (Size < sizeof(T)) {
      static_assert("Shouldn't happen");
    } else if constexpr (Size == sizeof(T)) {
      return toPack<TInput>(static_cast<TInput>(fromPack<T>(pack)));
    } else {
      BytePack<Size/Scale> ret;
      ret.half[0] = Converter<TInput, T>::downcast(pack.half[0]);
      ret.half[1] = Converter<TInput, T>::downcast(pack.half[1]);
      return ret;
    }
  }
};

// reduceCopy*() functions from common_kernel.h, modified for mixed precision and hardcoded for reduce-scatter
template<typename RedFn, typename T, typename TInput, int Unroll, int BytePerPack,
         int HasSrc1, int DstIsInput,
         typename IntBytes>
__device__ __forceinline__ void reduceCopyPacksMixedPrecision(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs,
    void* _src0, void* _src1, void* _dst,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  // src0 is always of type TInput
  // src1 (if present) is always of type T
  // dst might be of either type, depending on DstIsInput

  // *Bytes{Behind,Ahead} are calculated based on T

  constexpr size_t Scale = sizeof(T)/sizeof(TInput);
  static_assert(Scale == 2, "Currently, the data type should be exactly twice as large as the input type");

  // A hunk is the amount of contiguous data a warp consumes per loop iteration
  // assuming all threads partake.
  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // This thread's initial position.
  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  // Number of hunks to be consumed over all warps.
  IntBytes nHunksAhead = nBytesAhead/(BytePerHunk + !BytePerHunk);
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%(BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead%(BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t src0 = cvta_to_global(_src0) + threadBytesBehind/Scale;
  uintptr_t src1 = HasSrc1 ? cvta_to_global(_src1) + threadBytesBehind : 0;
  uintptr_t dst = cvta_to_global(_dst) + threadBytesBehind/(DstIsInput ? Scale : 1);

  // We dictate loop termination condition according to whether partial hunks
  // can be handled or not.
  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    RedFn preFn(preOpArgs[0]);

    BytePack<BytePerPack> acc[Unroll];
    #pragma unroll Unroll
    for (int u=0; u < Unroll; u++) {
      // Use volatile loads in case credits are polled for with volatile (instead of acquire).
      acc[u] = Converter<TInput, T>::upcast(ld_volatile_global<BytePerPack/Scale>(src0));
      acc[u] = applyPreOp(preFn, acc[u]);
      src0 += WARP_SIZE*BytePerPack/Scale;
    }

    if (HasSrc1) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Use volatile loads in case credits are polled for with volatile (instead of acquire).
        acc[u] = applyReduce(redFn, acc[u], ld_volatile_global<BytePerPack>(src1));
        src1 += WARP_SIZE*BytePerPack;
      }
    }

    if (DstIsInput) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++)
        acc[u] = applyPostOp(redFn, acc[u]);
    }

    #pragma unroll Unroll
    for (int u=0; u < Unroll; u++) {
      if (DstIsInput) {
        st_global<BytePerPack/Scale>(dst, Converter<TInput, T>::downcast(acc[u]));
        dst += WARP_SIZE*BytePerPack/Scale;
      } else {
        st_global<BytePerPack>(dst, acc[u]);
        dst += WARP_SIZE*BytePerPack;
      }
    }

    src0 += (nWarps-1)*BytePerHunk/Scale;
    src1 += (nWarps-1)*BytePerHunk;
    dst += (nWarps-1)*BytePerHunk/(DstIsInput ? Scale : 1);

    threadBytesBehind += nWarps*BytePerHunk;
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
  }

  // The last loop iteration could have been partial, i.e. not taken by all
  // threads. The threads that weren't included need an extra subtraction to
  // make the value warp uniform.
  if (Unroll==1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  // Rotate warps so the warp which got the least work here will be warp 0.
  // This effectively assigns: warp = (warp-nHunks+nWarps)%nWarps;
  thread = lane - nHunksAhead*WARP_SIZE;
}

template<int Unroll, typename RedFn, typename T, typename TInput,
         int HasSrc1, int DstIsInput,
         typename IntBytes>
__device__ __forceinline__ void reduceCopyMixedPrecision(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs,
    void* src0, void* src1, void* dst,
    IntBytes nElts
  ) {
  int lane = thread%WARP_SIZE;
  constexpr int BigPackSize = 16;

  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts*sizeof(T);

  if constexpr (BigPackSize > sizeof(T)) {
    // Check that all pointers are BigPackSize aligned.
    bool aligned = true;
    if (lane == 0) {
      aligned &= 0 == cvta_to_global(src0) % BigPackSize;
    }
    if (HasSrc1 && lane == 1) {
      aligned &= 0 == cvta_to_global(src1) % BigPackSize;
    }
    if (lane == 2) {
      aligned &= 0 == cvta_to_global(dst) % BigPackSize;
    }
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacksMixedPrecision<RedFn, T, TInput, Unroll, BigPackSize, HasSrc1, DstIsInput>
        (nThreads, /*&*/thread, redArg, preOpArgs,
         src0, src1, dst, /*&*/nBytesBehind, /*&*/nBytesAhead);
      if (nBytesAhead == 0) return;

      reduceCopyPacksMixedPrecision<RedFn, T, TInput, /*Unroll=*/1, BigPackSize,  HasSrc1, DstIsInput>
        (nThreads, /*&*/thread, redArg, preOpArgs,
         src0, src1, dst, /*&*/nBytesBehind, /*&*/nBytesAhead);
      if (nBytesAhead == 0) return;
    }
  }

  reduceCopyPacksMixedPrecision<RedFn, T, TInput, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T), HasSrc1, DstIsInput>
    (nThreads, /*&*/thread, redArg, preOpArgs,
     src0, src1, dst, /*&*/nBytesBehind, /*&*/nBytesAhead);
  if (nBytesAhead == 0) return;

  reduceCopyPacksMixedPrecision<RedFn, T, TInput, /*Unroll=*/1, /*BytePerPack=*/sizeof(T), HasSrc1, DstIsInput>
    (nThreads, /*&*/thread, redArg, preOpArgs,
     src0, src1, dst, /*&*/nBytesBehind, /*&*/nBytesAhead);
}

// A trimmed-down, ring-only, customized version of the primitives in `prims_simple.h`
template <typename T, typename TInput, typename RedOp>
class MixedPrecisionReduceScatterPrims {
private:
  static constexpr int SlicePerChunk = REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS;
  static constexpr int StepPerSlice = REDUCESCATTER_SLICESTEPS;

  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       OffsFifoEnabled = 0x80,
                       SizesFifoEnabled = 0x100,
                       ThreadsSynced = 0x800;
public:
  template <int Send, int Recv>
  __device__ __forceinline__ void doRound(int roundIndex) {
    constexpr int FirstRound = Send && !Recv;
    constexpr int LastRound = !Send && Recv;

    ssize_t srcIx = chunkOffset + ringRanks[nranks - roundIndex] * size;

    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    int slice = 0;
    int offset = 0;

    if (threadIdx.x < nworkers && offset < nelem) {
      #pragma unroll 1
      do {
        sliceSize = min(sliceSize, nelem - offset);

        if (flags & RoleInput) {
          ncclShmem.groups[0].srcs[0] = userBuff + srcIx + offset;
        }
        if (LastRound && (flags & RoleOutput)) {
          ncclShmem.groups[0].dsts[0] = userBuff + chunkOffset + offset;
        }

        waitPeer<Send, Recv>(sliceSize);
        subBarrier();

        // `prims_simple.h` tries to avoid doing unnecessary reduceCopy() if we are already aborted,
        // but we don't really mind doing some extra work.
        auto& group = ncclShmem.groups[0];
        reduceCopyMixedPrecision<
          ncclCollUnroll(), RedOp, T, TInput,
          !FirstRound /*HasSrc1*/, LastRound /*DstIsInput*/
        >(
          threadIdx.x, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs,
          group.srcs[0], group.srcs[1], group.dsts[0],
          sliceSize
        );

        barrier();
        postPeer<Send, Recv>(sliceSize > 0);
        offset += sliceSize;
        ++slice;
      } while (slice < SlicePerChunk && offset < nelem);
    }

    #pragma unroll 1
    while (slice < SlicePerChunk) {
      sliceSize = min(sliceSize, nelem - offset);

      waitPeer<Send, Recv>(0);

      barrier();
      postPeer<Send, Recv>(sliceSize > 0);
      offset += sliceSize;
      ++slice;
    }
  }

  __device__ __forceinline__ void prepareRound(ssize_t chunkOffset, int nelem) {
    this->chunkOffset = chunkOffset;
    this->nelem = nelem;
  }

  __device__ MixedPrecisionReduceScatterPrims(int nthreads, const void* inputBuf, void* outputBuf,
                                              uint64_t redOpArg, ncclRing* ring,
                                              ssize_t stepSize, int nranks, ssize_t size)
    : nthreads(nthreads)
    , nworkers(nthreads - WARP_SIZE)
    , ringRanks(ring->userRanks)
    , stepSize(stepSize)
    , nranks(nranks)
    , size(size)
  {
    constexpr int ThreadsPerSync = 8;

    int tid = threadIdx.x;

    flags = 0;
    // Preserve the indexes from `prims_simple.h` for simplicity
    if (tid == 0) {
      flags |= RoleWaitRecv;
    } else if (tid == 1) {
      flags |= RoleInput;
    } else if (tid == ThreadsPerSync) {
      flags |= RoleWaitSend;
    } else if (tid == ThreadsPerSync + 1) {
      flags |= RoleOutput;
    } else if (tid == nthreads - 2 * ThreadsPerSync) {
      flags |= RolePostRecv;
    } else if (tid == nthreads - ThreadsPerSync) {
      flags |= RolePostSend;
    }

    loadConn(ring->prev, RolePostRecv, RoleWaitRecv, true /*isRecv*/);
    loadConn(ring->next, RolePostSend, RoleWaitSend, false /*isRecv*/);

    if (flags & RoleInput) {
      ncclShmem.redOpArgs[0] = redOpArg;
      userBuff = (TInput*)inputBuf;
    }
    if (flags & RoleOutput) {
      userBuff = (TInput*)outputBuf;
    }
  }

  __device__ ~MixedPrecisionReduceScatterPrims() {
    if (!(flags & ThreadsSynced)) {
      barrier();
    }

    if (flags & (RolePostSend | RolePostRecv)) {
      auto& group = ncclShmem.groups[0];
      ((flags & RolePostSend) ? group.sendConns : group.recvConns)[0]->step = step;
    }

    barrier();
  }

private:
  __device__ void barrier() {
    flags |= ThreadsSynced;
    asm volatile("bar.sync %0, %1;" :: "r"(15), "r"(nthreads) : "memory");
  }

  __device__ void subBarrier() {
    asm volatile("bar.sync %0, %1;" :: "r"(8), "r"(nworkers) : "memory");
  }

  template <int Send, int Recv>
  __device__ __forceinline__ void waitPeer(int nelts) {
    const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    if (flags & (Recv * RoleWaitRecv | Send * RoleWaitSend)) {
      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        connStepCache = ld_volatile_global(connStepPtr);
        // Check for kernel abort.
        spins++;
        if (!(flags & Aborted) && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
          if (*ncclShmem.comm.abortFlag) {
            flags |= Aborted;
            ncclShmem.aborted = 1;
          }
          spins = 0;
        }
        if (flags & Aborted) {
          break;
        }
      }

      if (isSendNotRecv && (flags & SizesFifoEnabled)) {
        connSizesFifoPtr[step % NCCL_STEPS] = nelts * sizeof(T);
      }

      const int curStep = step % NCCL_STEPS;
      const int delta =
        (flags & OffsFifoEnabled)
          ? loadInt(connOffsFifoPtr + curStep)/sizeof(T)
          : curStep*stepSize;
      auto* buf = connEltsFifo + delta;
      if (isSendNotRecv) {
        ncclShmem.groups[0].dsts[0] = buf;
      } else {
        ncclShmem.groups[0].srcs[1] = buf;
      }

      step += StepPerSlice;
    }
  }

  template <int Send, int Recv>
  __device__ void postPeer(bool dataStored) {
    if (flags & (Recv * RolePostRecv | Send * RolePostSend)) {
      step += StepPerSlice;
      if (Send && (flags & RolePostSend) && dataStored) {
        fence_acq_rel_sys();
      }
      st_relaxed_sys_global(connStepPtr, step);
    }
  }

  __device__ __forceinline__ void loadConn(int peerIndex, int postRole, int waitRole, bool isRecv) {
    if (flags & (postRole | waitRole)) {
      auto& peer = ncclShmem.channel.peers[peerIndex];
      auto* conn = isRecv ? peer->recv : peer->send;
      step = conn->step;
      step = roundUp(step, SlicePerChunk * StepPerSlice);
      if (flags & postRole) {
        if (isRecv) {
          connStepPtr = conn->head;
          *connStepPtr = step;
        } else {
          connStepPtr = conn->tail;
        }
      }
      if (flags & waitRole) {
        auto& group = ncclShmem.groups[0];
        ((isRecv) ? group.recvConns : group.sendConns)[0] = conn;
        connStepPtr = isRecv ? conn->tail : conn->head;
        connStepCache = ld_volatile_global(connStepPtr);

        if (conn->offsFifo != nullptr) {
          flags |= OffsFifoEnabled;
          connOffsFifoPtr = conn->offsFifo;
        }
        connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];

        if (!isRecv) {
          if (conn->sizesFifo != nullptr) {
            flags |= SizesFifoEnabled;
            connSizesFifoPtr = conn->sizesFifo;
          }
        }
      }
    }
  }

  const int nthreads;
  const int nworkers;

  void** sendBufs;
  const size_t* sendLens;
  const int* ringRanks;

  const int stepSize;
  const int nranks;

  const ssize_t size;

  int nelem;
  int flags;

  ssize_t chunkOffset;

  uint64_t step;
  int *connOffsFifoPtr;
  union {
    TInput *userBuff;
    T *connEltsFifo;
  };
  int volatile *connSizesFifoPtr;
  uint64_t *connStepPtr;
  uint64_t connStepCache;
};


// A trimmed-down, customized version of `reduce_scatter.h`
template <typename T, typename TInput, typename RedOp>
struct RunWorkElementMixedPrecision<ncclFuncMixedPrecisionReduceScatter, T, TInput, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ void run(ncclWorkElem* args) {
    assert(args->nWarps > 1); // we need an extra warp of non-workers

    const int nthreads = args->nWarps * WARP_SIZE;
    const int nChannels = args->nChannels;
    const int bid = args->bid;
    const int nranks = ncclShmem.comm.nRanks;

    // `ProtoSimple::calcBytePerStep()` is inlined here
    const ssize_t stepSize = (ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS) / sizeof(T);
    const ssize_t chunkSize = stepSize * REDUCESCATTER_CHUNKSTEPS;
    const ssize_t loopSize = args->nChannels * chunkSize;
    const ssize_t size = args->count;

    MixedPrecisionReduceScatterPrims<T, TInput, RedOp>
      prims(nthreads, args->sendbuff, args->recvbuff, args->redOpArg, &ncclShmem.channel.ring, stepSize, nranks, size);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize = min(chunkSize, divUp(size - gridOffset, nChannels));
      // Account for the extra warp not reducing any data
      realChunkSize = roundUp(realChunkSize, (nthreads - WARP_SIZE) * sizeof(uint64_t) / sizeof(T));

      const ssize_t chunkOffset = gridOffset + bid * realChunkSize;
      const int nelem = static_cast<int>(min(realChunkSize, size - chunkOffset));

      prims.prepareRound(chunkOffset, nelem);

      // SENDBUF -> NETBUF
      prims.doRound<1 /*Send*/, 0 /*Recv*/>(1);
      for (int j = 2; j < nranks; ++j) {
        // SENDBUF + NETBUF -> NETBUF
        prims.doRound<1 /*Send*/, 1 /*Recv*/>(j);
      }
      // SENDBUF + NETBUF -> SENDBUF
      prims.doRound<0 /*Send*/, 1 /*Recv*/>(nranks);
    }
  }
};
