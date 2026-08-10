//
//  DemonstrationData.swift
//  umi_day_iphone
//
//  Created by Austin Patel on 9/12/24.
//  Copyright © 2024 Apple. All rights reserved.
//

import Foundation
import simd
import CoreVideo
import CoreMedia
import Speech

extension simd_float4x4: Codable {
    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        // Decode each row individually (instead of columns)
        let row0 = try container.decode(SIMD4<Float>.self)
        let row1 = try container.decode(SIMD4<Float>.self)
        let row2 = try container.decode(SIMD4<Float>.self)
        let row3 = try container.decode(SIMD4<Float>.self)
        // Convert rows into columns (row-major to column-major conversion)
        self.init(
            SIMD4<Float>(row0.x, row1.x, row2.x, row3.x),
            SIMD4<Float>(row0.y, row1.y, row2.y, row3.y),
            SIMD4<Float>(row0.z, row1.z, row2.z, row3.z),
            SIMD4<Float>(row0.w, row1.w, row2.w, row3.w)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        // Convert columns into rows for row-major encoding
        let row0 = SIMD4<Float>(columns.0.x, columns.1.x, columns.2.x, columns.3.x)
        let row1 = SIMD4<Float>(columns.0.y, columns.1.y, columns.2.y, columns.3.y)
        let row2 = SIMD4<Float>(columns.0.z, columns.1.z, columns.2.z, columns.3.z)
        let row3 = SIMD4<Float>(columns.0.w, columns.1.w, columns.2.w, columns.3.w)
        // Encode each row individually
        try container.encode(row0)
        try container.encode(row1)
        try container.encode(row2)
        try container.encode(row3)
    }
}

class TaskSegmentation {
    var taskStart: Date?
    var taskEnd: Date?
    var name: String?
    
    init() {
        
    }
    
    init(taskStart: Date, taskEnd: Date, name: String? = nil) {
        self.taskStart = taskStart
        self.taskEnd = taskEnd
        self.name = name
    }
}

enum DemonstrationType: String, Codable {
    case QRCalibration = "qr_calibration"
    case GripperCalibration = "gripper_calibration"
    case Demonstration = "demonstration"
}

enum DemonstrationLabelType: String, Codable {
    case Narration
    case Predefined
    case None
    case GripperWidth
}

enum DemonstrationSaveType: String, Codable, CaseIterable {
    case JSON
    case RGB
    case UltrawideRGB
    case DepthMap
    case DepthPreviewMap
    case DepthConfidenceMap
}

class DemonstrationData : Encodable {
//    parameters we save in json
    var poseTimes: [String] = []
    var rgbTimes: [String] = []
    var ultrawideRGBTimes: [String] = []
    var depthTimes: [String] = []
    var poseTransforms: [simd_float4x4] = []
    var recordingStartTime: String
    var side: String = ""
    var timeIPhoneAheadOfQRinMS: Int
    var type: DemonstrationType
    var qrCalibrationRunName: String
    var gripperCalibrationRunName: String
    var sessionName: String
    var note: String
    var hasRGB: Bool
    var hasUltrawideRGB: Bool
    var hasDepth: Bool
    var taskStartTimestamps: [String] = []
    var taskEndTimestamps: [String] = []
    var taskNames: [String] = []
    var narrationStartTimestamps: [String] = []
    var narrationEndTimestamps: [String] = []
    var narrationTexts: [String] = []
    var labelType: DemonstrationLabelType
    var taskID: String
    var hasGoPro: Bool
    var hasAudio: Bool = false
    
//    parameters we do not save in json
    var recordingName: String
    var rgbVideoWriter: VideoWriter?
    var ultrawideRGBVideoWriter: VideoWriter?
    var depthVideoWriter: DepthVideoWriter?
    var depthPreviewVideoWriter: DepthPreviewVideoWriter?
    var depthConfidenceVideoWriter: DepthConfidenceVideoWriter?
    var frameCount: Int = 0
    
    private enum CodingKeys : String, CodingKey {
//        these are the keys that are saved to .json
//        note that all timestamps saved to the JSON file are AFTER QR latency adjustment (so no need to apply it again later)
        case poseTimes
        case rgbTimes
        case ultrawideRGBTimes
        case depthTimes
        case poseTransforms
        case recordingStartTime
        case side
        case timeIPhoneAheadOfQRinMS
        case type
        case qrCalibrationRunName
        case gripperCalibrationRunName
        case sessionName
        case note
        case hasRGB
        case hasUltrawideRGB
        case hasDepth
        case frameCount
        case taskStartTimestamps
        case taskEndTimestamps
        case taskNames
        case narrationStartTimestamps
        case narrationEndTimestamps
        case narrationTexts
        case labelType
        case taskID
        case hasGoPro
        case hasAudio
    }
    
    init(recordingName: String, isRight: Bool, recordingStartTime: Date, timeIPhoneAheadOfQRinMS: Int, demonstrationType: DemonstrationType, qrCalibrationRunName: String, gripperCalibrationRunName: String, sessionName: String, note: String, hasGoPro: Bool, labelType: DemonstrationLabelType, taskID: String?) {
        //timeARaheadOfQRinMS: if positive it means that ARkit time is ahead of QR code time -> need to subtract this latency to fix it!
        side = isRight ? "right" : "left"
        self.recordingName = recordingName
        self.type = demonstrationType
        self.timeIPhoneAheadOfQRinMS = (self.type == .Demonstration && hasGoPro) ? timeIPhoneAheadOfQRinMS : 0 // latency is 0 if we are doing a demonstration recording RGB from iPhone
        self.qrCalibrationRunName = self.type == .Demonstration ? qrCalibrationRunName : ""
        self.gripperCalibrationRunName = self.type == .Demonstration ? gripperCalibrationRunName : ""
        self.sessionName = sessionName
        self.note = note
        self.hasGoPro = hasGoPro
        self.hasRGB = !hasGoPro && demonstrationType == .Demonstration
        self.hasUltrawideRGB = !hasGoPro
        self.hasDepth = !hasGoPro && demonstrationType == .Demonstration
        self.labelType = labelType
        
        if let taskID {
            self.taskID = taskID
        } else {
            self.taskID = ""
        }
        
        // start recording time
        self.recordingStartTime = "" // need to do this for next line to work
        self.recordingStartTime = DateManager.getISOFormatter().string(from: adjustDueToQRLatency(recordingStartTime))
    }
    
    func logAudio(audioSampleBuffer: CMSampleBuffer) {
        if type == .Demonstration {
            ultrawideRGBVideoWriter?.appendAudio(sampleBuffer: audioSampleBuffer)
            rgbVideoWriter?.appendAudio(sampleBuffer: audioSampleBuffer)
            hasAudio = true
        }
    }
    
    func logFrame(pose: simd_float4x4, poseTime: Date, rgb: CVPixelBuffer, depthMap: CVPixelBuffer?, depthConfidenceMap: CVPixelBuffer?, ultrawidergb: CVPixelBuffer?, arkitTimestamp: TimeInterval) {
        let poseTime = DateManager.getISOFormatter().string(from: adjustDueToQRLatency(poseTime))
        let arkitTimeString = String(format: "%.5f", arkitTimestamp)
        let frameTime = CMTimeMake(value: Int64(arkitTimestamp*10000), timescale: 10000) // trick to represent float as fraction
        
        // setup the RGB video writer if this is the first frame
        if self.hasRGB && rgbVideoWriter == nil {
            do {
                let rgbOutputUrl = try self.getURL(demonstrationSaveType: .RGB)
                self.rgbVideoWriter = try VideoWriter(outputURL: rgbOutputUrl, width: CVPixelBufferGetWidth(rgb), height: CVPixelBufferGetHeight(rgb), includeAudio: true)
                self.rgbVideoWriter!.startWriting(at: frameTime)
            } catch {
                print("Failed to open video writer for RGB")
            }
        }
        
        if self.hasUltrawideRGB && ultrawideRGBVideoWriter == nil, let ultrawidergb = ultrawidergb {
            do {
                let rgbOutputUrl = try self.getURL(demonstrationSaveType: .UltrawideRGB)
                self.ultrawideRGBVideoWriter = try VideoWriter(outputURL: rgbOutputUrl, width: CVPixelBufferGetWidth(ultrawidergb), height: CVPixelBufferGetHeight(ultrawidergb), includeAudio: true)
                self.ultrawideRGBVideoWriter!.startWriting(at: frameTime)
            } catch {
                print("Failed to open video writer for ultrawide RGB")
            }
        }
        
        if depthMap == nil {
            hasDepth = false
        }
        
        // setup the depth video writer if this is the first frame
        if self.hasDepth && depthVideoWriter == nil { // some devices don't have LiDAR so depthMap can be nil
            do {
                let depthOutputUrl = try self.getURL(demonstrationSaveType: .DepthMap)
                self.depthVideoWriter = try DepthVideoWriter(outputURL: depthOutputUrl, width: CVPixelBufferGetWidth(depthMap!), height: CVPixelBufferGetHeight(depthMap!))
                
                let depthPreviewOutputUrl = try self.getURL(demonstrationSaveType: .DepthPreviewMap)
                self.depthPreviewVideoWriter = try DepthPreviewVideoWriter(outputURL: depthPreviewOutputUrl, width: CVPixelBufferGetWidth(depthMap!), height: CVPixelBufferGetHeight(depthMap!))
                self.depthPreviewVideoWriter!.startWriting(at: frameTime)
                
                let depthConfidenceOutputUrl = try self.getURL(demonstrationSaveType: .DepthConfidenceMap)
                self.depthConfidenceVideoWriter = try DepthConfidenceVideoWriter(outputURL: depthConfidenceOutputUrl, width: CVPixelBufferGetWidth(depthConfidenceMap!), height: CVPixelBufferGetHeight(depthConfidenceMap!))
                self.depthConfidenceVideoWriter!.startWriting(at: frameTime)
            } catch {
                print("Failed to open video writer for depth")
            }
        }
        
        if self.hasRGB {
            // save the RGB image
//            print("Image dimensions: W: \(CVPixelBufferGetWidth(rgb)) H: \(CVPixelBufferGetHeight(rgb))")
            rgbVideoWriter!.appendVideo(pixelBuffer: rgb, at: frameTime)
            rgbTimes.append(poseTime)
        }
        
        if self.hasUltrawideRGB {
            if let ultrawidergb = ultrawidergb {
                ultrawideRGBVideoWriter!.appendVideo(pixelBuffer: ultrawidergb, at: frameTime)
                ultrawideRGBTimes.append(poseTime)
            } else {
                ultrawideRGBTimes.append("")
            }
        }
        
        if self.hasDepth {
            depthVideoWriter!.append(pixelBuffer: depthMap!)
            depthPreviewVideoWriter!.append(pixelBuffer: depthMap!, at: frameTime)
            depthConfidenceVideoWriter!.append(pixelBuffer: depthConfidenceMap!, at: frameTime)
            depthTimes.append(poseTime)
        }
        
        // only need to save pose times and transforms if it's a demonstration
        if type == .Demonstration {
            poseTimes.append(poseTime)
            poseTransforms.append(pose)
        }
        self.frameCount += 1
    }
        
    private static func getFolderURL() throws -> URL {
        let folderURL = try FileManager.default.url(for: .documentDirectory,
                                                    in: .userDomainMask,
                                                    appropriateFor: nil,
                                                    create: false).appendingPathComponent("demonstration_data")
        
        if !FileManager.default.fileExists(atPath: folderURL.path) {
            try FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: true, attributes: nil)
        }
        
        return folderURL
    }
    
    public static func getURL(recordingName: String, demonstrationSaveType: DemonstrationSaveType) throws -> URL {
        let folderURL = try getFolderURL()
        
        switch demonstrationSaveType {
        case .JSON:
            return folderURL.appendingPathComponent("\(recordingName).json")
        case .RGB:
            return folderURL.appendingPathComponent("\(recordingName)_rgb.mp4")
        case .UltrawideRGB:
            return folderURL.appendingPathComponent("\(recordingName)_ultrawidergb.mp4")
        case .DepthMap:
            return folderURL.appendingPathComponent("\(recordingName)_depth.raw")
        case .DepthPreviewMap:
            return folderURL.appendingPathComponent("\(recordingName)_depthpreview.mp4")
        case .DepthConfidenceMap:
            return folderURL.appendingPathComponent("\(recordingName)_depthconfidence.mp4")
        }
        
    }
    
    public func getURL(demonstrationSaveType: DemonstrationSaveType) throws -> URL {
        return try DemonstrationData.getURL(recordingName: recordingName, demonstrationSaveType: demonstrationSaveType)
    }
    
    public static func hasDataType(recordingName: String, demonstrationSaveType: DemonstrationSaveType) -> Bool {
        do {
            let fileURL = try DemonstrationData.getURL(recordingName: recordingName, demonstrationSaveType: demonstrationSaveType)
            let fileManager = FileManager.default
            return fileManager.fileExists(atPath: fileURL.path())
        } catch {
            return false
        }
    }
    
    private func adjustDueToQRLatency(_ date: Date) -> Date {
        let offsetInterval = TimeInterval(Double(-timeIPhoneAheadOfQRinMS) / 1000) // negative is very important! because we are fixing this latency
        let adjustedDate = date.addingTimeInterval(offsetInterval)
        //        print("time: \(date.timeIntervalSince1970) adjustedtime: \(adjustedTime.timeIntervalSince1970)  qrLatencyMilliseconds: \(qrLatencyMilliseconds) offsetInterval: \(offsetInterval)")\
        return adjustedDate
    }
    
    func setFinalData(timeIPhoneAheadOfQRinMS: Int?, speechRecognitionResult: SFSpeechRecognitionResult?, transcriptionStartTime: Date?, taskSegmentationEvents: [TaskSegmentation]) {
        // timeIPhoneAheadOfQRinMS is only required if this a QR calibration
        // speechRecognitionResult is only required if this is a demonstration and we are in narration mode
        
        if type == .QRCalibration {
            // in this case we update the QR calibration so that it reflects the new value in the saved file
            self.timeIPhoneAheadOfQRinMS = timeIPhoneAheadOfQRinMS!
        }
        
        var segmentationEvents = taskSegmentationEvents
        
        if type == .Demonstration {
            // if we do predefined tasks, then segmentationEvents is already fully ready to go
            // if we do language narration, then the segmentationEvents won't have the name or start time properties set (only the end time will be set), so we need to determine the start time and name based on the narration data
            
            // apply QR latency adjustment to segmentation events
            taskSegmentationEvents.forEach { event in
                if event.taskStart != nil {
                    event.taskStart = adjustDueToQRLatency(event.taskStart!)
                }
                if event.taskEnd != nil {
                    event.taskEnd = adjustDueToQRLatency(event.taskEnd!)
                }
            }
            
            // segment out subtasks from narration and segmentation events
            if labelType == .Narration {
                if speechRecognitionResult != nil {
                    // pull out the data from the speech result
                    var narrationStartTimestamps: [Date] = []
                    var narrationEndTimestamps: [Date] = []
                    
                    speechRecognitionResult?.bestTranscription.segments.forEach { segment in
                        // compute time offset including shift to UTC time at start of recording and then shift to QR code
                        let adjustedTranscriptionStartTime = adjustDueToQRLatency(transcriptionStartTime!)
                        let relativeStartTime = segment.timestamp
                        let textStartDate = adjustedTranscriptionStartTime.addingTimeInterval(relativeStartTime)
                        let textEndDate = textStartDate.addingTimeInterval(segment.duration)
                        
                        narrationStartTimestamps.append(textStartDate)
                        narrationEndTimestamps.append(textEndDate)
                        
                        self.narrationStartTimestamps.append(DateManager.getISOFormatter().string(from: textStartDate))
                        self.narrationEndTimestamps.append(DateManager.getISOFormatter().string(from: textEndDate))
                        narrationTexts.append(segment.substring)
                    }
                    
                    // segment out the subtasks from the narration data
                    var segmentStartIndex = 0
                    var subtaskIndex = 0
                    for index in narrationTexts.indices {
                        // if the word "done" is said and it's the start of a segment, then just move to the next word and cut the word "done" from the segment
                        if NarrationCommands.isDoneWord(narrationTexts[index]) && segmentStartIndex == index {
                            segmentStartIndex += 1
                            continue
                        }
                        
                        let currentEndOfNarration = narrationEndTimestamps[index]
                        var isSubtaskDone = false
                        if index < narrationTexts.count - 1 {
                            // the narration times will set the end time of the previous segment to match the start time of the current segment if they are part of a contiguous command; any different between them means the last command has since ended and there was a gap before this word, thus the previous task is done. Sometimes small breaks are still introduced, so require at least half a second gap before considering a task done
                            let startOfNextNarration = narrationStartTimestamps[index+1]
                            isSubtaskDone = startOfNextNarration.timeIntervalSince(currentEndOfNarration) > 0.5
                            
                            // if a done word is next even if 0.5 seconds haven't passed, then we consider the subtask to be done. If the done word is said right after the start of the narration (no break), then just discard this subtask because it will have no data in it since it was stopped right away
                            if NarrationCommands.isDoneWord(narrationTexts[index+1]) {
                                if startOfNextNarration.timeIntervalSince(currentEndOfNarration) == 0 {
                                    // no time past between when the task label ended and when the stop word started, so just cancel this subtask
                                    segmentStartIndex = index + 1
                                    continue
                                } else {
                                    // some time passed before the stop word was used, so even if it wasn't 0.5 seconds, consider the subtask to to be done
                                    isSubtaskDone = true
                                }
                            }
                        } else {
                            isSubtaskDone = true // this was the last language narration
                        }
                        
                        if isSubtaskDone {
                            // we have found a segment from startSegmentIndex to index, inclusive and we know the exact start time
                            
                            // the challenge is figuring out the exact end time. this can be from multiple sources
                            // source 1 is checking if there is a segmentation event with a taskEnd time that is AFTER currentEnd (i.e., right after the last word of the language annotation is spoken) and before the start of the next language annotation. There is a segmentation event for the end of the episode as well that can be used
                            // source 2 is if there is no segmentation events satisfying the requirement above, then we consider the end to be at the start of the next language label
                            
                            // we simplify all this logic, by just inserting a segmentation event at the start of each narration event and ignoring segmentation events if their end time is before the current narration's start time
                            
                            if index < narrationTexts.count - 1 {
                                // we know startOfNextNarration is part of a the next subtask label (not part of the current narration label)
                                let startOfNextNarration = narrationStartTimestamps[index+1]
                                
                                // insert a segmentation event ending at nextStart
                                let endNarrationEvent = TaskSegmentation()
                                endNarrationEvent.taskEnd = startOfNextNarration
                                
                                // insert the event in sorted order
                                var insertionIndex = subtaskIndex
                                while insertionIndex <= segmentationEvents.count {
                                    if insertionIndex == segmentationEvents.count {
                                        segmentationEvents.append(endNarrationEvent)
                                        break
                                    } else {
                                        if segmentationEvents[insertionIndex].taskEnd!.timeIntervalSince(endNarrationEvent.taskEnd!) > 0 { // meaning the entry currently there is later than the insertion value
                                            segmentationEvents.insert(endNarrationEvent, at: insertionIndex)
                                            break
                                        } else {
                                            insertionIndex += 1
                                        }
                                    }
                                }
                            }
                            
                            // loop through segmentation events until we have one that has an end time that is after the start time
                            while(segmentationEvents[subtaskIndex].taskEnd!.timeIntervalSince(currentEndOfNarration) < 0) {
                                segmentationEvents.remove(at: subtaskIndex)
                            }
                            
                            segmentationEvents[subtaskIndex].taskStart = currentEndOfNarration
                            segmentationEvents[subtaskIndex].name = narrationTexts[segmentStartIndex...index].joined(separator: " ")
                            segmentStartIndex = index + 1
                            subtaskIndex += 1
                        }
                    }
                    while segmentationEvents.count > subtaskIndex {
                        segmentationEvents.removeLast()
                    }
                    
                    assert(subtaskIndex == segmentationEvents.count)
                } else {
                    // in this case narration mode was enabled, but no narration was detected, so cut the end task event
                    segmentationEvents = []
                }
            }
            
            // convert all of the segmentation events into the format stored in this class
            segmentationEvents.forEach { event in
                let startTime = DateManager.getISOFormatter().string(from: event.taskStart!)
                let endTime = DateManager.getISOFormatter().string(from: event.taskEnd!)
                let taskName = event.name!
                
                assert (event.taskEnd!.timeIntervalSince(event.taskStart!) > 0) // end has to be after start
                
                taskStartTimestamps.append(startTime)
                taskEndTimestamps.append(endTime)
                taskNames.append(taskName)
            }
        }
    }
    
    func saveLocally() throws {
        // saves demonstration data to local storage
        let encoder = JSONEncoder()
        let data = try encoder.encode(self)
        let outfile = try getURL(demonstrationSaveType: .JSON)
        try data.write(to: outfile)
        
        if hasRGB {
            rgbVideoWriter!.finishWriting {
                let outURL = try? self.getURL(demonstrationSaveType: .RGB)
                print("Video writing finished: \(outURL!)")
            }
        }
        
        if hasUltrawideRGB {
            ultrawideRGBVideoWriter!.finishWriting {
                let outURL = try? self.getURL(demonstrationSaveType: .UltrawideRGB)
                print("Ultrawide video writing finished: \(outURL!)")
            }
        }
        
        if hasDepth {
            depthVideoWriter!.finishWriting()
            let outURL = try? self.getURL(demonstrationSaveType: .DepthMap)
            print("Video writing finished: \(outURL!)")
            
            depthPreviewVideoWriter!.finishWriting {
                let outURL = try? self.getURL(demonstrationSaveType: .DepthPreviewMap)
                print("Video writing finished: \(outURL!)")
            }
            depthConfidenceVideoWriter!.finishWriting {
                let outURL = try? self.getURL(demonstrationSaveType: .DepthConfidenceMap)
                print("Video writing finished: \(outURL!)")
            }
        }
    }
    
    static func saveExternally(recordingName: String, directoryURL: URL) throws {
        // copies the local save of this demonstration data to another location
        // should only be called after saveLocally is called
        for dataType in DemonstrationSaveType.allCases {
            if Self.hasDataType(recordingName: recordingName, demonstrationSaveType: dataType) {
                let file = try DemonstrationData.getURL(recordingName: recordingName, demonstrationSaveType: dataType)
                let destination = directoryURL.appendingPathComponent(file.lastPathComponent)
                try FileManager.default.copyItem(at: file, to: destination)
            }
        }
    }
    
    static func discard(recordingName: String) throws {
        // Iterating over data types saved
        for saveType in DemonstrationSaveType.allCases {
            let rmfile = try Self.getURL(recordingName: recordingName, demonstrationSaveType: saveType)
            if FileManager.default.fileExists(atPath: rmfile.path()) {
                try FileManager.default.removeItem(at: rmfile)
            }
        }
    }
    
    static func discardDemonstrationsDir() throws {
        let rmdir = try Self.getFolderURL()
        if FileManager.default.fileExists(atPath: rmdir.path()) {
            try FileManager.default.removeItem(at: rmdir)
        }
    }
    
    static func listDemonstrations() throws -> [String] {
        // Get the URL for the DocumentDirectory
        let documentsURL = try getFolderURL()
        
        // Retrieve the contents of the DocumentDirectory
        let documentsPath = documentsURL.path
        let files = try FileManager.default.contentsOfDirectory(atPath: documentsPath)
        
        // Filter to include only JSON files
        let jsonFiles = files.filter { $0.hasSuffix(".json") }
        
        // Remove ".json" extension and sort the names
        var strippedFileNames = jsonFiles.map { $0.replacingOccurrences(of: ".json", with: "") }
        strippedFileNames = strippedFileNames.sorted().reversed()
        
        return strippedFileNames
    }
    
    static func loadAsString(recordingName: String) throws -> String {
        let infile = try Self.getURL(recordingName: recordingName, demonstrationSaveType: .JSON)
        return try String(contentsOf: infile, encoding: .utf8)
    }
    
}
