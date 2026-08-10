//
//  DepthVideoWriter.swift
//  umi_day_iphone
//
//  Created by Austin Patel on 12/20/24.
//  Copyright © 2024 Apple. All rights reserved.
//

//
//  VideoWriter.swift
//  umi_day_iphone
//
//  Created by Austin Patel on 12/19/24.
//  Copyright © 2024 Apple. All rights reserved.
//

import AVFoundation
import Accelerate

class DepthConfidenceVideoWriter {
    private var assetWriter: AVAssetWriter!
    private var assetWriterInput: AVAssetWriterInput!
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor!
    private var isWritingStarted = false

    init(outputURL: URL, width: Int, height: Int) throws {
        // Initialize AVAssetWriter
        assetWriter = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        
        // Define video settings
        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
        
        // Create AVAssetWriterInput
        assetWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        assetWriterInput.expectsMediaDataInRealTime = true
        assetWriter.add(assetWriterInput)
        
        // Create Pixel Buffer Adaptor with the new pixel format (kCVPixelFormatType_OneComponent32Float)
        pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: assetWriterInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_OneComponent32Float, // Updated format
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ]
        )
    }
    
    func startWriting(at time: CMTime) {
        assetWriter.startWriting()
        assetWriter.startSession(atSourceTime: time)
        isWritingStarted = true
    }
    
    func append(pixelBuffer: CVPixelBuffer, at time: CMTime) {
        guard isWritingStarted, assetWriterInput.isReadyForMoreMediaData else { return }
        
        // Convert the pixel buffer to kCVPixelFormatType_OneComponent32Float
        if let convertedBuffer = convertTo32Float(pixelBuffer: pixelBuffer) {
            pixelBufferAdaptor.append(convertedBuffer, withPresentationTime: time)
        }
    }
    
    func finishWriting(completion: @escaping () -> Void) {
        assetWriterInput.markAsFinished()
        assetWriter.finishWriting {
            completion()
        }
    }
    
    private func convertTo32Float(pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        // Get the pixel buffer attributes and data size
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // Create a new pixel buffer with kCVPixelFormatType_OneComponent32Float
        var newPixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent32Float,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height
        ]
        
        let status = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_OneComponent32Float, attrs as CFDictionary, &newPixelBuffer)
        guard status == kCVReturnSuccess, let newBuffer = newPixelBuffer else {
            print("Error creating new pixel buffer")
            return nil
        }
        
        // Lock the base address of the original and new pixel buffers
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(newBuffer, [])
        
        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        let newBaseAddress = CVPixelBufferGetBaseAddress(newBuffer)
        
        // Get the pixel buffer data stride and count
        let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pixelCount = width * height
        
        // Perform the conversion using vDSP
        let srcPointer = baseAddress!.assumingMemoryBound(to: UInt8.self)
        let dstPointer = newBaseAddress!.assumingMemoryBound(to: Float.self)
        
        // Use vDSP to convert the pixel data (8-bit to 32-bit float)
        vDSP_vfltu8(srcPointer, 1, dstPointer, 1, vDSP_Length(pixelCount))
        
        // Normalize values to the range [0.0, 1.0]
        vDSP_vsdiv(dstPointer, 1, [255.0], dstPointer, 1, vDSP_Length(pixelCount))
        
        // Unlock the base addresses
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferUnlockBaseAddress(newBuffer, [])
        
        return newBuffer
    }
}

