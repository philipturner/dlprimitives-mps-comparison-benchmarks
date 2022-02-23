import MetalPerformanceShaders

guard CommandLine.arguments.count == 5,
      CommandLine.arguments[1] == "-gflops",
      Double(CommandLine.arguments[2]) != nil,
      CommandLine.arguments[3] == "-bps",
      Double(CommandLine.arguments[4]) != nil else {
    print("""
    Usage:
        swift mps_flops.swift -gflops MAX_GFLOPS -bps MAX_GB/S
    """)
    exit(0)
}

let ref_flops = Double(CommandLine.arguments[2])! * 1_000_000_000
let ref_bps = Double(CommandLine.arguments[4])! * 1_000_000_000
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

struct Metrics {
    var flops: Double
    var bps: Double
}

func test_gemm(
    warms: Int,
    calcs: Int,
    m: Int,
    n: Int,
    k: Int,
    ta: Bool,
    tb: Bool
) -> Metrics {
    func makeMatrix(rows: Int, cols: Int) -> MPSMatrix {
        let numBytes = rows * cols * MemoryLayout<Float>.stride
        #if os(macOS) && !arch(arm64)
        let storageMode: MTLResourceOptions = .storageModeManaged
        #else
        let storageMode: MTLResourceOptions = .storageModeShared
        #endif
        let buffer = device.makeBuffer(length: numBytes, options: storageMode)!
        
        let desc = MPSMatrixDescriptor(
            rows: rows,
            columns: cols,
            rowBytes: cols * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        return MPSMatrix(buffer: buffer, descriptor: desc)
    }
    
    let A = makeMatrix(rows: (ta == false) ? m : k, cols: (ta == false) ? k : m)
    let B = makeMatrix(rows: (tb == false) ? k : n, cols: (tb == false) ? n : k)
    let C = makeMatrix(rows: m, cols: n)
    
    let kernel = MPSMatrixMultiplication(
        device: device,
        transposeLeft: ta,
        transposeRight: tb,
        resultRows: m,
        resultColumns: n,
        interiorColumns: k,
        alpha: 1,
        beta: 0
    )
    
    func runGEMM() -> MTLCommandBuffer {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        kernel.encode(
            commandBuffer: commandBuffer,
            leftMatrix: A,
            rightMatrix: B,
            resultMatrix: C
        )
        commandBuffer.commit()
        return commandBuffer
    }
    
    for _ in 0..<warms {
        _ = runGEMM()
    }
    
    var seconds: Double = 0
    for _ in 0..<calcs {
        let commandBuffer = runGEMM()
        commandBuffer.waitUntilCompleted()
        seconds += commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    }
    
    let flop = m * n * (k * 2 - 1) * calcs
    let bytes = (m * k + k * n + m * n) * MemoryLayout<Float>.stride * calcs
    
    return Metrics(
        flops: Double(flop) / seconds,
        bps: Double(bytes) / seconds
    )
}

let setups: [(Int, Int, Int)] = [
    ( 512,   512,  512 ),
    ( 1024, 1024, 1024 ),
    ( 1025, 1025, 1025 ),
    ( 2048, 2048, 2048 ),
    ( 2049, 2049, 2049 ),
    ( 64,   2048,   64 ),
    ( 2048,   64, 2048 ),
    ( 2048, 2048,   64 ),
    ( 2048,   64,   64 ),
    ( 64,   2048, 2048 ),
    ( 64,     64, 2048 )
]

for ta in [false, true] {
    for tb in [false, true] {
        for i in 0..<setups.count {
            let (M, N, K) = setups[i]
            
            let approx_flop = Double(M * N * K * 2)
            let gemm_per_sec = approx_flop / ref_flops
            let target_calls = 2 / gemm_per_sec
            
            let calls = max(5, min(200, Int(target_calls)))
            let warms = max(1, calls / 5)
            
            let m = test_gemm(
                warms: warms, calcs: calls,
                m: M, n: N, k: K,
                ta: ta, tb: tb
            )
            print(
                String(format: "  \(ta ? "T" : "N")\(tb ? "T" : "N") %2d: %4d, %4d, %4d ",
                i, M, N, K)
            )
            
            let flops_per = m.flops / ref_flops * 100
            let bandw_per = m.bps / ref_bps * 100
            let max_per = max(flops_per, bandw_per)
            let limited = (flops_per > bandw_per) ? "gflops" : "memory"
            
            let gf_raw = String(format: "%8.1f", m.flops * 1e-9)
            let gf_per = String(format: "%5.2f", flops_per)
            let bw_raw = String(format: "%8.1f", m.bps * 1e-9)
            let bw_per = String(format: "%5.2f", flops_per)
            let lim_per = String(format: "%5.2f", max_per)
            
            print(
                String(format: "  \(gf_raw) GFlops (\(gf_per)%%) \(bw_raw) GB/s (\(bw_per)%%) limited by \(limited) \(lim_per)%%\n")
            )
        }
    }
}
