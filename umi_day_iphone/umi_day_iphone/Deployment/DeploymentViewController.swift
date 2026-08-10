//
//  DeploymentViewController.swift
//  umi_day_iphone
//
//  Created by Austin Patel on 1/31/25.
//  Copyright © 2025 Apple. All rights reserved.
//

import UIKit
import AVFoundation

class RgbSocketPacket {
    var rgb: CIImage
    
    init(rgb: CIImage) {
        self.rgb = rgb
    }
    
    func toEncodedString() -> String {
        return SocketUtil.encodePixelBufferToPNGBase64(rgb)!
    }
}

class DepthSocketPacket {
    var depth: CVPixelBuffer
    
    init(depth: CVPixelBuffer) {
        self.depth = depth
    }
    
    func toEncodedString() -> String {
        return SocketUtil.encodeDepthPixelBufferToBase64(depth)!
    }
}

class DeploymentViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureDepthDataOutputDelegate {
    @IBOutlet weak var enablePreviewSwitch: UISwitch!
    @IBOutlet weak var enableDepthSwitch: UISwitch!
    @IBOutlet weak var resetSocketButton: UIButton!
    
    @IBOutlet weak var lidarView: UIView!
    @IBOutlet weak var ultrawideView: UIView!
    @IBOutlet weak var wideView: UIView!
    
    @IBOutlet weak var stopCameraButton: UIButton!
    @IBOutlet weak var resetCameraButton: UIButton!
    
    var lidarImageView: UIImageView!
    
    let captureSession = AVCaptureMultiCamSession()
    
    // Define inputs
    var lidarCamera: AVCaptureDevice?
    var ultraWideCamera: AVCaptureDevice?
    var wideCamera: AVCaptureDevice?
    
    var lidarPreviewLayer: AVCaptureVideoPreviewLayer?
    var ultrawidePreviewLayer: AVCaptureVideoPreviewLayer?
    var widePreviewLayer: AVCaptureVideoPreviewLayer?
    
    // Define outputs
    var wideOutput: AVCaptureVideoDataOutput?
    var ultrawideOutput: AVCaptureVideoDataOutput?
    var depthOutput: AVCaptureDepthDataOutput?
    
    // Socket
    let mainSocketClient = SocketClient()
    let ultrawideSocketClient = SocketClient()
    let depthSocketClient = SocketClient()
    var hostIP: String = "192.168.2.18"
    var mainPort: Int = 5555
    var ultrawidePort: Int = 5556
    var depthPort: Int = 5557
    var socketTimer: Timer?
    var stopPressed: Bool = false
    var previewEnabled: Bool = false
    var depthEnabled: Bool = false
    
    // FPS counter
    @IBOutlet weak var ultrawideFPSCoutner: UILabel!
    @IBOutlet weak var wideFPSCounter: UILabel!
    @IBOutlet weak var depthFPSCounter: UILabel!
    var lastUltrawideTime: CMTime? = nil
    var lastWideTime: CMTime? = nil
    var lastDepthTime: CMTime? = nil
    
    // Socket status
    @IBOutlet weak var ultrawideSocketStatusLabel: UILabel!
    @IBOutlet weak var wideSocketStatusLabel: UILabel!
    @IBOutlet weak var depthSocketStatusLabel: UILabel!
    
    @IBOutlet weak var depthCameraHeaderLabel: UILabel!
    
    override func viewDidLoad() {
        // prevent screen from going to sleep
        UIApplication.shared.isIdleTimerDisabled = true
        
        // check if camera preview enabled
        let defaults = UserDefaults.standard
        var deployEnablePreview = defaults.object(forKey: "deployEnablePreview") as? Bool
        if deployEnablePreview == nil {
            deployEnablePreview = enablePreviewSwitch.isOn
            defaults.set(deployEnablePreview!, forKey: "deployEnablePreview")
        }
        enablePreviewSwitch.isOn = deployEnablePreview!
        previewEnabled = deployEnablePreview!
        
        // check if depth enabled
        var deployEnableDepth = defaults.object(forKey: "deployEnableDepth") as? Bool
        if deployEnableDepth == nil {
            deployEnableDepth = enableDepthSwitch.isOn
            defaults.set(deployEnableDepth!, forKey: "deployEnableDepth")
        }
        enableDepthSwitch.isOn = deployEnableDepth!
        depthEnabled = deployEnableDepth!
        
        // Create UIImageView for LiDAR depth data
        if previewEnabled && depthEnabled {
            lidarImageView = UIImageView(frame: lidarView.bounds)
            lidarImageView.contentMode = .scaleAspectFill
            lidarView.addSubview(lidarImageView)
        }
        
        if !depthEnabled {
            depthCameraHeaderLabel.text = ""
        }
        
        // Hide FPS counters initially
        ultrawideFPSCoutner.text = ""
        wideFPSCounter.text = ""
        depthFPSCounter.text = ""
        
        // Hide socket text initially
        ultrawideSocketStatusLabel.text = ""
        wideSocketStatusLabel.text = ""
        depthSocketStatusLabel.text = ""
        
        socketTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            if !self.stopPressed {
                self.validateSocketConnection()
            }
            
            // update labels if camera feed gets stuck
            let uptime: Double = CACurrentMediaTime()
            if let lastUltrawideTime = self.lastUltrawideTime {
                let delta = uptime - CMTimeGetSeconds(lastUltrawideTime)
                if delta > 1 { // 1 second has past since last frame
                    self.ultrawideFPSCoutner.text = "CAMERA NOT UPDATING"
                }
            }
            
            if let lastWideTime = self.lastWideTime {
                let delta = uptime - CMTimeGetSeconds(lastWideTime)
                if delta > 1 { // 1 second has past since last frame
                    self.wideFPSCounter.text = "CAMERA NOT UPDATING"
                }
            }
            
            if let lastDepthTime = self.lastDepthTime {
                let delta = uptime - CMTimeGetSeconds(lastDepthTime)
                if delta > 1 { // 1 second has past since last frame
                    self.depthFPSCounter.text = "CAMERA NOT UPDATING"
                }
            }
            
        }
    }
    
    override func viewDidAppear(_ animated: Bool) {
        self.validateSocketConnection()
        configureCaptureSession()
    }
    
    private func configureCaptureSession() {
        captureSession.beginConfiguration()
        
        guard AVCaptureMultiCamSession.isMultiCamSupported else {
            print("Multi-camera capture is not supported on this device.")
            return
        }
        
        var deviceTypes: [AVCaptureDevice.DeviceType] = [.builtInUltraWideCamera]
        if depthEnabled {
            deviceTypes.append(.builtInLiDARDepthCamera) // includes wide angle as part of depth
        } else {
            deviceTypes.append(.builtInWideAngleCamera)
        }
        
        // Get LiDAR (includes wide RGB) and Ultrawide Camera devices
        let deviceDiscovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: .video,
            position: .back
        )

        for device in deviceDiscovery.devices {
            switch device.deviceType {
            case .builtInLiDARDepthCamera:
                lidarCamera = device
                wideCamera = device // wide angle tied into LiDAR device
            case .builtInUltraWideCamera:
                ultraWideCamera = device
            case .builtInWideAngleCamera:
                wideCamera = device
            default:
                break
            }
        }
        
        // add camera input/outputs
        do {
            // Configure Ultra Wide Camera
            if let ultrawideDevice = ultraWideCamera {
                // input (for some reason this has to go before doing the configuration, otherwise the 60fps setting doensn't work and you get 24FPS)
                let ultrawideInput = try AVCaptureDeviceInput(device: ultrawideDevice)
                if captureSession.canAddInput(ultrawideInput) {
                    captureSession.addInput(ultrawideInput)
                }
                
                // set resolution and frame rate
                try ultrawideDevice.lockForConfiguration()
                if let format = ultrawideDevice.formats.first(where: { format in
                    if !format.isMultiCamSupported || format.isVideoHDRSupported { // see comment for wide camera for why we add this check for HDR (for some reason it's related to whether we can change focus modes)
                        return false
                    }
                    let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                    let supportedFrameRates = format.videoSupportedFrameRateRanges
                    
                    return dimensions.width == 640 && dimensions.height == 480 && supportedFrameRates.contains { $0.maxFrameRate >= 60 } // lowest supported resolution that supports multicam
                }) {
                    ultrawideDevice.activeFormat = format
                    ultrawideDevice.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 60)
                    ultrawideDevice.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 60)
                    
                    // it appears ARKit has locked focus for ultrawide so we lock it also here
                    if ultrawideDevice.isFocusModeSupported(.locked) {
                        ultrawideDevice.focusMode = .locked
                    }
                    ultrawideDevice.setFocusModeLocked(lensPosition: 0.7529412) // here we use the number retrieved by printing out the lens position of the ultrawide camera in ARKit on iPhone 15 Pro. It appears to be fixed at this one lens position
                } else {
                    assert(false)
                }
                assert(ultrawideDevice.focusMode == .locked)
                ultrawideDevice.unlockForConfiguration()
                
                // preview output
                if previewEnabled {
                    ultrawidePreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
                    ultrawidePreviewLayer?.videoGravity = .resizeAspect
                    ultrawidePreviewLayer?.connection?.videoRotationAngle = 0
                    ultrawidePreviewLayer?.frame = ultrawideView.bounds
                    ultrawideView.layer.addSublayer(ultrawidePreviewLayer!)
                }
                
                // delegate output
                ultrawideOutput = AVCaptureVideoDataOutput()
                if captureSession.canAddOutput(ultrawideOutput!) {
                    captureSession.addOutput(ultrawideOutput!)
                    ultrawideOutput!.setSampleBufferDelegate(self, queue: DispatchQueue(label: "ultrawideOutputQueue"))
                }
            }
            
            // Setup main RGB (if using depth then main RGB is tied into LiDAR device
            if let wideDevice = wideCamera {
                // input (for some reason this has to go before doing the configuration, otherwise the 60fps setting doensn't work and you get 24FPS or 30FPS)
                let wideInput = try AVCaptureDeviceInput(device: wideDevice)
                if captureSession.canAddInput(wideInput) {
                    captureSession.addInput(wideInput)
                }
                
                // set resolution and frame rate
                try wideDevice.lockForConfiguration()
                if let format = wideDevice.formats.first(where: { format in
                    if !format.isMultiCamSupported || format.isVideoHDRSupported { // for some bizarre reason I found that the formats that support HDR do not have auto focus working... (likely this is just a correlation, but adding condition to skip formats that have HDR enabled me to ensure the auto focus system still works...
                        return false
                    }
                    let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                    let supportedFrameRates = format.videoSupportedFrameRateRanges
                    
                    let valid = dimensions.width == 640 && dimensions.height == 480 && supportedFrameRates.contains { $0.maxFrameRate >= 60 } // lowest supported resolution that supports multicam
                    return valid
                }) {
                    wideDevice.activeFormat = format
                    wideDevice.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 60)
                    wideDevice.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 60)
                    
                    if depthEnabled {
                        wideDevice.activeDepthDataMinFrameDuration = CMTime(value: 1, timescale: 30) // note you can only get depth at 30fps when streaming due to AV limitation (even though ARKit gives you 60fps depth)
                    }
                } else {
                    assert(false)
                }
                assert(wideDevice.focusMode == .continuousAutoFocus) // ARKit uses continuous auto focus for main camera
                wideDevice.unlockForConfiguration()

                // preview output
                if previewEnabled {
                    widePreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
                    widePreviewLayer?.videoGravity = .resizeAspect
                    widePreviewLayer?.connection?.videoRotationAngle = 0
                    widePreviewLayer?.frame = wideView.bounds
                    wideView.layer.addSublayer(widePreviewLayer!)
                }
                
                // delegate output
                wideOutput = AVCaptureVideoDataOutput()
                if captureSession.canAddOutput(wideOutput!) {
                    captureSession.addOutput(wideOutput!)
                    wideOutput!.setSampleBufferDelegate(self, queue: DispatchQueue(label: "wideOutputQueue"))
                }
            }
            
            // Configure LiDAR Camera
            if depthEnabled {
                if let lidarDevice = lidarCamera {
                    // add output delegate for depth
                    depthOutput = AVCaptureDepthDataOutput()
                    if captureSession.canAddOutput(depthOutput!) {
                        captureSession.addOutput(depthOutput!)
                        depthOutput!.setDelegate(self, callbackQueue: DispatchQueue(label: "depthOutputQueue"))
                    }
                }
            }
            
            // start running
            captureSession.commitConfiguration()
            captureSession.startRunning()
        } catch {
            print("Error configuring capture session: \(error)")
        }
    }
    
    func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CIImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let transform = CGAffineTransform(scaleX: CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
                                          y: CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer)))
        let resizedImage = ciImage.transformed(by: transform)
        return resizedImage
    }
    
    // Delegate method to process captured frames
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if output == ultrawideOutput {
//            print("Captured from ultrawide camera \(sampleBuffer.outputPresentationTimeStamp) \(sampleBuffer.presentationTimeStamp)")
            if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                let origWidth = CVPixelBufferGetWidth(imageBuffer)
                let origHeight = CVPixelBufferGetHeight(imageBuffer)
                let newWidth = 320
                let newHeight = 240
                
                let resized = resizePixelBuffer(imageBuffer, width: newWidth, height: newHeight)!

//                print("ultrawide Output dimensions: \(origWidth)x\(origHeight) -> \(newWidth)x\(newHeight)")
                
                let dataPacket = RgbSocketPacket(rgb: resized)
                if ultrawideSocketClient.isConnected() {
                    ultrawideSocketClient.sendData(dataPacket.toEncodedString(), channel: "rgb")
                }
                
                // Update FPS counter
                var fps = ""
                let curTimestamp = sampleBuffer.presentationTimeStamp
                if let lastUltrawideTime = lastUltrawideTime {
                    fps = String(format: "%.2f", 1.0 / Double(CMTimeGetSeconds(curTimestamp - lastUltrawideTime))) + " FPS"
                }
                lastUltrawideTime = curTimestamp
                
                DispatchQueue.main.async {
                    self.ultrawideFPSCoutner.text = fps
                    self.ultrawideSocketStatusLabel.text = "Socket: " + self.ultrawideSocketClient.getStatus()
                }
            }
        } else if output == wideOutput {
//            print("Captured from wide camera")
            if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                let origWidth = CVPixelBufferGetWidth(imageBuffer)
                let origHeight = CVPixelBufferGetHeight(imageBuffer)
                let newWidth = 320
                let newHeight = 240
                
                let resized = resizePixelBuffer(imageBuffer, width: newWidth, height: newHeight)!

//                print("wide Output dimensions: \(origWidth)x\(origHeight) -> \(newWidth)x\(newHeight)")
                
                let dataPacket = RgbSocketPacket(rgb: resized)
                if mainSocketClient.isConnected() {
                    mainSocketClient.sendData(dataPacket.toEncodedString(), channel: "rgb")
                }
                
                // Update FPS counter
                var fps = ""
                let curTimestamp = sampleBuffer.presentationTimeStamp
                if let lastWideTime = lastWideTime {
                    fps = String(format: "%.2f", 1.0 / Double(CMTimeGetSeconds(curTimestamp - lastWideTime))) + " FPS"
                }
                lastWideTime = curTimestamp
                
                DispatchQueue.main.async {
                    self.wideFPSCounter.text = fps
                    self.wideSocketStatusLabel.text = "Socket: " + self.mainSocketClient.getStatus()
                }
            }
        } else {
            assert(false)
        }
    }
    
    // Delegate method to process LiDAR depth data
    func depthDataOutput(_ output: AVCaptureDepthDataOutput, didOutput depthData: AVDepthData, timestamp: CMTime, connection: AVCaptureConnection) {
//        print("Captured from LiDAR")
        
        assert(depthEnabled)
        
        // streaming
        if depthSocketClient.isConnected() {
            print("depth socket connected so sending!")
            let convertedDepthData = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
            let pixelBuffer = convertedDepthData.depthDataMap
            let dataPacket = DepthSocketPacket(depth: pixelBuffer)
            depthSocketClient.sendData(dataPacket.toEncodedString(), channel: "depth")
        } else {
            print("depth socket not connected so not sending")
        }
        
        // fps counter
        var fps = ""
        if let lastDepthTime = lastDepthTime {
            fps = String(format: "%.2f", 1.0 / Double(CMTimeGetSeconds(timestamp - lastDepthTime))) + " FPS"
        }
        lastDepthTime = timestamp
        
        DispatchQueue.main.async {
            self.depthFPSCounter.text = fps
            self.depthSocketStatusLabel.text = "Socket: " + self.depthSocketClient.getStatus()
        }
        
        // visual preview
        if previewEnabled {
            let imageBuffer = depthData.depthDataMap
            let depthImage = self.depthDataToImage(depthData)
            DispatchQueue.main.async {
                self.lidarImageView.image = depthImage
            }
        }
    }
    
    @IBAction func collectDemonstrationButtonPress(_ sender: Any) {
        let defaults = UserDefaults.standard
        defaults.set("demonstration", forKey: "appMode")
        
        // Present message saying to restart app
        let alertController = UIAlertController(title: "Warning", message: "Close and reopen the app to enter demonstration mode", preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
            alertController.dismiss(animated: true, completion: nil)
            // do nothing
        }
        alertController.addAction(okAction)
        self.present(alertController, animated: true, completion: nil)
    }
    
    func depthDataToImage(_ depthData: AVDepthData) -> UIImage? {

        var convertedDepthData = depthData
        if depthData.depthDataType != kCVPixelFormatType_DisparityFloat32 {
            convertedDepthData = depthData.converting(toDepthDataType: kCVPixelFormatType_DisparityFloat32)
        }
        
        let depthMap = convertedDepthData.depthDataMap
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        
        let buffer = DepthPreviewVideoWriter.convertDepthBufferToOneComponent32Float(pixelBuffer: depthMap)!
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let context = CIContext()
        
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            return UIImage(cgImage: cgImage)
        }
        
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
        return nil
    }
    
    func validateSocketConnection() {
        // if the socket is connected, then validate that connection by sending a message to the server and waiting for an ack
        // if the socket is not connected, then attempt connection
        if mainSocketClient.isConnected() {
            mainSocketClient.validateConnection()
        } else {
            mainSocketClient.connect(hostIP: hostIP, hostPort: mainPort)
        }
        if ultrawideSocketClient.isConnected() {
            ultrawideSocketClient.validateConnection()
        } else {
            ultrawideSocketClient.connect(hostIP: hostIP, hostPort: ultrawidePort)
        }
        if depthSocketClient.isConnected() {
            depthSocketClient.validateConnection()
        } else {
            depthSocketClient.connect(hostIP: hostIP, hostPort: depthPort)
        }
    }
    
    func disconnectSocket() {
        mainSocketClient.disconnect()
        ultrawideSocketClient.disconnect()
    }
    
    @IBAction func resetSocketButtonPress(_ sender: Any) {
        disconnectSocket()
        validateSocketConnection()
        stopPressed = false
    }
    
    @IBAction func stopSocketButtonPress(_ sender: Any) {
        disconnectSocket()
        stopPressed = true
    }
    
    @IBAction func enablePreviewSwitchToggled(_ sender: Any) {
        let defaults = UserDefaults.standard
        defaults.set(enablePreviewSwitch.isOn, forKey: "deployEnablePreview")
        
        // Present message saying to restart app
        let alertController = UIAlertController(title: "Warning", message: "Close and reopen the app to enable/disable camera preview", preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
            alertController.dismiss(animated: true, completion: nil)
            // do nothing
        }
        alertController.addAction(okAction)
        self.present(alertController, animated: true, completion: nil)
    }
    
    @IBAction func enableDepthSwitchToggled(_ sender: Any) {
        let defaults = UserDefaults.standard
        defaults.set(enableDepthSwitch.isOn, forKey: "deployEnableDepth")
        
        // Present message saying to restart app
        let alertController = UIAlertController(title: "Warning", message: "Close and reopen the app to enable/disable depth streaming", preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
            alertController.dismiss(animated: true, completion: nil)
            // do nothing
        }
        alertController.addAction(okAction)
        self.present(alertController, animated: true, completion: nil)
    }
    
    @IBAction func stopCameraButtonPressed(_ sender: Any) {
        captureSession.stopRunning()
    }
    
    @IBAction func resetCameraButtonPressed(_ sender: Any) {
        captureSession.stopRunning()
        captureSession.startRunning()
    }
}
