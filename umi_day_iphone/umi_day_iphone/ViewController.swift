/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Main view controller for the AR experience. UMIFT.
*/

import UIKit
import RealityKit
import ARKit
import VisionKit
import MultipeerConnectivity
import Speech
import ObjectiveC
import MediaPlayer
import Foundation
import AVKit

class DataPacket {
    var transformMatrix: simd_float4x4
    var timestamp: Double

    init(transformMatrix: simd_float4x4, timestamp: Double) {
        self.transformMatrix = transformMatrix
        self.timestamp = timestamp
    }
    func toBytes() -> Data {
        var data = Data()

        // Append pose data
        for i in 0..<4 {
            for j in 0..<4 {
                var val = transformMatrix[i][j]
                data.append(Data(bytes: &val, count: MemoryLayout<Float>.size))
            }
        }

        // Append timestamp
        var timestampVal = timestamp
        data.append(Data(bytes: &timestampVal, count: MemoryLayout<Int64>.size))

        return data
    }
}


extension ARFrame {
    func getCapturedUltraWideImage() -> CVPixelBuffer? {
        // magic to get access to ultrawide image
        let ivarName = "_capturedUltraWideImage"

        guard let ivar = class_getInstanceVariable(ARFrame.self, ivarName) else {
            print("Failed to find ivar: \(ivarName)")
            return nil
        }

        let ptr = UnsafeRawPointer(Unmanaged.passUnretained(self).toOpaque()).advanced(by: ivar_getOffset(ivar))
        let pixelBuffer = ptr.load(as: CVPixelBuffer?.self)

        return pixelBuffer
    }
    
    func getUltraWideTimestamp() -> TimeInterval? {
        // magic to get access to ultrawide timestamp
        let ivarName = "_ultraWideImageTimestamp"

        guard let ivar = class_getInstanceVariable(ARFrame.self, ivarName) else {
            print("Failed to find ivar: \(ivarName)")
            return nil
        }
        
        let ptr = UnsafeRawPointer(Unmanaged.passUnretained(self).toOpaque()).advanced(by: ivar_getOffset(ivar))
        let timestamp = ptr.load(as: Double?.self)

        return timestamp
    }
    
    func getUltraWideCamera() -> ARCamera? {
        // magic to get access to ultrawide camera device
        let ivarName = "_ultraWideCamera"

        guard let ivar = class_getInstanceVariable(ARFrame.self, ivarName) else {
            print("Failed to find ivar: \(ivarName)")
            return nil
        }
        
        let ptr = UnsafeRawPointer(Unmanaged.passUnretained(self).toOpaque()).advanced(by: ivar_getOffset(ivar))
        let arCamera = ptr.load(as: ARCamera?.self)

        return arCamera
    }
}

struct Constants {
    static let readQRIntervalStartValue: Int = 5
}

class RecordingMessage : NSObject, NSSecureCoding {
    var startRecording: Bool
    var recordingName: NSString?
    var recordingStartTime: NSDate?
    
    init(startRecording: Bool, recordingName: NSString?, recordingStartTime: NSDate?) {
        self.startRecording = startRecording
        self.recordingName = recordingName
        self.recordingStartTime = recordingStartTime
    }
    
    // Indicate that secure coding is supported
    static var supportsSecureCoding: Bool {
        return true
    }

    // Initializer to decode data
    required init?(coder decoder: NSCoder) {
        // Decode your properties here
        // Ensure you're using `decodeObject(of:forKey:)` to ensure secure decoding
        startRecording = decoder.decodeBool(forKey: "startRecording")
        recordingName = decoder.decodeObject(forKey: "recordingName") as? NSString
        recordingStartTime = decoder.decodeObject(forKey: "recordingStartTime") as? NSDate
    }

    // Method to encode data
    func encode(with coder: NSCoder) {
        // Encode your properties here
        // You can use methods like `aCoder.encode(someProperty, forKey: "someKey")`
        coder.encode(startRecording, forKey: "startRecording")
        coder.encode(recordingName, forKey: "recordingName")
        coder.encode(recordingStartTime, forKey: "recordingStartTime")
    }
}

enum RecordingMode {
    case single
    case both
    case none
}

class DemonstrationTasksState {
    var currentTaskIndex: Int = 0
    var tasks: [String] = []
    var currentlyRecordingTask: Bool = false
    var taskSegmentationEvents: [TaskSegmentation] = []
    var labelType: DemonstrationLabelType = .None
    
    init() {
        reset()
    }
    
    func reset() {
        currentTaskIndex = 0
        tasks = []
        currentlyRecordingTask = false
        taskSegmentationEvents = []
    }
}

class ViewController: UIViewController, ARSessionDelegate, GoProControllerDelegate {
    
    @IBOutlet weak var messageLabel: MessageLabel!
    
    @IBOutlet weak var resetPoseButton: UIButton!
    @IBOutlet weak var poseLabel: UILabel!
    @IBOutlet weak var stateLabel: UILabel!
    @IBOutlet weak var streamPeerSocketSwitch: UISwitch!
    @IBOutlet weak var recordingModeIcon: UIImageView!
    @IBOutlet weak var goProConnectedIcon: UIImageView!
    @IBOutlet weak var leftRightSegmentedControl: UISegmentedControl!
    @IBOutlet weak var recordTypeSegmentedControl: UISegmentedControl!
    @IBOutlet weak var nameSessionButton: UIButton!
    @IBOutlet weak var noteButton: UIButton!
    @IBOutlet weak var useGoProSwitch: UISwitch!
    @IBOutlet weak var useViewerSwitch: UISwitch!
    @IBOutlet weak var currentTaskLabel: UILabel!
    @IBOutlet weak var labelTypeSegmentedControl: UISegmentedControl!
    @IBOutlet weak var recordButton: UIButton!
    @IBOutlet weak var nextTaskButton: UIButton!
    @IBOutlet weak var tasksButton: UIButton!
    @IBOutlet weak var demosButton: UIButton!
    @IBOutlet weak var goProButton: UIButton!
    @IBOutlet weak var micView: UIImageView!
    
    @IBOutlet var arViewHolder: UIView!
    var arView: ARView? // exists only if enabled by user
    @objc var session: ARSession = ARSession()
    
    var multipeerSession: MultipeerSession?
    
    var coachingOverlay: ARCoachingOverlayView?
    
    // A dictionary to map MultiPeer IDs to ARSession ID's.
    // This is useful for keeping track of which peer created which ARAnchors.
    var peerSessionIDs = [MCPeerID: String]()
    
    var sessionIDObservation: NSKeyValueObservation?
    
    var configuration: ARWorldTrackingConfiguration?
    
    var prevTimestampThisDevice: Double = 0.0
    var prevTimestampOtherDevice: Double = 0.0
    private var publishPose: Bool = false
    let socketClient = SocketClient()
    var hostIP: String = "192.168.2.18"
    var hostPort: Int = 5555
    var isHostSide: Bool = true
    var isRightSide: Bool = true
    
    var streamPeerSocket: Bool = false
    var dataScanner: DataScannerViewController?
    
    // world anchor
    var justAddedWorldOrigin: Bool = false
    var worldAnchorInitialCountdown: Int = 60
    var worldAnchorCountdown: Int = -1
    var worldAnchor: ARAnchor? = nil
    
    // local anchor
    var localAnchorInitialCountdown: Int = 60
    var localAnchorCountdown: Int = -1
    var localWorldAnchor: ARAnchor? = nil
    
    var isRecording: Bool = false
    var recordingName: String = ""
    var recordingStartTime: Date?
    var demonstrationData: DemonstrationData?
    var arKitTimeOffset: Double = 0
    
    var streamingPeerSessionId: String? = nil
    
    var recordingMode: RecordingMode = .none
    
    var isStartLoggingQR: Bool = false
    var readQRInterval: Int = Constants.readQRIntervalStartValue // read QR every X frames
    
    var useGoPro: Bool = false
    var useViewer: Bool = false
    
    var entireDemoSpeechRecognizer: SpeechRecognizer?
//    var liveSpeechRecognizer: SpeechRecognizer?
    
    var tasksState: DemonstrationTasksState = DemonstrationTasksState()
    
    var micInitiallyFound: Bool = false
    var micCurrentlyConnected: Bool = false
    
    private var eventInteraction: AVCaptureEventInteraction?
    
    var fps: Double = 0
        
    override func viewDidLoad() {
        poseLabel.text = ""
        stateLabel.text = ""
        
        // set defaults if not already set
        let defaults = UserDefaults.standard
        var oldCorrection = defaults.object(forKey: "timeIPhoneAheadOfQRinMS") as? Int
        if oldCorrection == nil {
            defaults.set(0, forKey: "timeIPhoneAheadOfQRinMS")
        }
        var qrCalibrationRunName = defaults.object(forKey: "qrCalibrationRunName") as? String
        if qrCalibrationRunName == nil {
            defaults.set("", forKey: "qrCalibrationRunName")
        }
        var gripperCalibrationRunName = defaults.object(forKey: "gripperCalibrationRunName") as? String
        if gripperCalibrationRunName == nil {
            defaults.set("", forKey: "gripperCalibrationRunName")
        }
        var sessionName = defaults.object(forKey: "sessionName") as? String
        if sessionName == nil {
            defaults.set("", forKey: "sessionName")
        }
        var note = defaults.object(forKey: "note") as? String
        if note == nil {
            defaults.set("", forKey: "note")
        }
        var tasks = defaults.object(forKey: "tasks") as? [String]
        if tasks == nil {
            defaults.set([], forKey: "tasks")
        }
        var gripperWidthTaskID = defaults.object(forKey: "gripperWidthTaskID") as? String
        if gripperWidthTaskID == nil {
            defaults.set("", forKey: "gripperWidthTaskID")
        }
        var labelSelectedSegmentID = defaults.object(forKey: "labelSelectedSegmentID") as? Int
        if labelSelectedSegmentID == nil {
            labelSelectedSegmentID = 0
            defaults.set(0, forKey: "labelSelectedSegmentID")
        }
        labelTypeSegmentedControl.selectedSegmentIndex = labelSelectedSegmentID!
                
        // Remember state of use GoPro switch
        var useGoPro = defaults.object(forKey: "useGoPro") as? Bool
        if useGoPro == nil {
            useGoPro = useGoProSwitch.isOn
            defaults.set(useGoPro, forKey: "useGoPro")
        }
        useGoProSwitch.isOn = useGoPro!
        setUseGoPro(isOn: useGoPro!)
        
        // setup AR view and session        
        var useViewer = defaults.object(forKey: "useViewer") as? Bool
        if useViewer == nil {
            useViewer = useViewerSwitch.isOn
            defaults.set(useViewer, forKey: "useViewer")
        }
        useViewerSwitch.isOn = useViewer!
        self.useViewer = useViewer!
        
        // setup left/right
        var isRight = defaults.object(forKey: "isRight") as? Bool
        if isRight == nil {
            isRight = leftRightSegmentedControl.selectedSegmentIndex == 1
            defaults.set(isRight, forKey: "isRight")
        }
        leftRightSegmentedControl.selectedSegmentIndex = isRight! ? 1 : 0
        isRightSide = isRight!
        
        initialSetPreferredMicToHeadset()
        
        // init AR configuration
        configuration = ARWorldTrackingConfiguration()
        
        // record audio
        configuration?.providesAudioData = true
        
        // if using the viewer then initialize an ARView
        if useViewer! {
            arView = ARView(frame: view.bounds)
            arView!.session = session
            arViewHolder.insertSubview(arView!, at: 0)
        }
        
        updateTaskUI()
        
        // initialize speech detectors
//        liveSpeechRecognizer = SpeechRecognizer(shouldReportPartialResults: true, callback: narrationCallback)
//        liveSpeechRecognizer!.startTranscribing()
        
        initializeSession()
        
        configureHardwareInteraction()
    }
    
    func initialSetPreferredMicToHeadset() {
        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true)

            if let availableInputs = audioSession.availableInputs {
                for input in availableInputs {
                    if input.portType == .headsetMic {
//                        try audioSession.setPreferredInput(input)
                        micInitiallyFound = true
                        micCurrentlyConnected = true
                        messageLabel.displayMessage("🎤 Contact mic connected!")
                        onMicStateUpdate()
                        return
                    }
                }
            }
            
            messageLabel.displayMessage("Headset microphone not found.")
        } catch {
            messageLabel.displayMessage("Error setting headset microphone: \(error.localizedDescription)")
        }
        
        onMicStateUpdate()
    }
    
    private func configureHardwareInteraction() {
        // Create a new capture event interaction with a handler that captures a photo.
        let interaction = AVCaptureEventInteraction { [weak self] event in
            // Capture a photo on "press up" of a hardware button.
            if event.phase == .ended {
                self!.recordButtonPress()
            }
        }
        // Add the interaction to the view controller's view.
        view.addInteraction(interaction)
        eventInteraction = interaction
    }
    
    private var gopro: GoPro?
    private let goProManager: GoProManager = GoProManager()

    override func viewDidAppear(_ animated: Bool) {
        
        super.viewDidAppear(animated)

        session.delegate = self

        // Turn off ARView's automatically-configured session
        // to create and set up your own configuration.
        arView?.automaticallyConfigureSession = false

        // Enable a collaborative session.
        configuration?.isCollaborationEnabled = true
        
        // Enable realistic reflections.
        configuration?.environmentTexturing = .automatic
        
        // Enable the sceneDepth frame semantics
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration!.frameSemantics.insert(.sceneDepth)
            print("added .sceneDepth from AR configuration")
        } else {
            print("Scene depth is not supported on this device.")
        }

        // Begin the session.
        session.run(configuration!)
        
        // Use key-value observation to monitor your ARSession's identifier.
        sessionIDObservation = observe(\.session.identifier, options: [.new]) { object, change in
            print("SessionID changed to: \(change.newValue!)")
            
            // Tell all other peers about your ARSession's changed ID, so
            // that they can keep track of which ARAnchors are yours.
            guard let multipeerSession = self.multipeerSession else { return }
            self.sendARSessionIDTo(peers: multipeerSession.connectedPeers)
            self.computeARKitTimeOffset()
        }
        
        if useViewer {
            coachingOverlay = ARCoachingOverlayView()
            setupCoachingOverlay()
        }
        
        // Prevent the screen from being dimmed to avoid interrupting the AR experience.
        UIApplication.shared.isIdleTimerDisabled = true
        
        if useGoPro {
            reconnectGoPro()
        }
        
        computeARKitTimeOffset()
        
        // subscribe to headphone connected events
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(audioRouteChanged),
            name: AVAudioSession.routeChangeNotification,
            object: nil
        )
    }
    
    @objc private func audioRouteChanged(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else {
            return
        }
        
        switch reason {
        case .newDeviceAvailable, .oldDeviceUnavailable:
            checkForHeadsetMic()
        default:
            break
        }
    }
    
    private func checkForHeadsetMic() {
        let session = AVAudioSession.sharedInstance()
        let inputs = session.availableInputs ?? []
        
        for input in inputs {
            if input.portType == .headsetMic {
                do {
//                    try session.setPreferredInput(input)
                    if micInitiallyFound {
                        if !micCurrentlyConnected {
                            messageLabel.displayMessage("🎤 Contact mic connected!")
                        }
                    } else if !micCurrentlyConnected{
                        messageLabel.displayMessage("Found contact mic, but you must start the app with the contact mic connected for it to work")
                    }
                    micCurrentlyConnected = true
                } catch {
                    messageLabel.displayMessage("Failed to set headset mic to preferred input!")
                    micCurrentlyConnected = false
                }
                
                onMicStateUpdate()
                
                return
            }
        }
        if micCurrentlyConnected {
            messageLabel.displayMessage("Contact mic disconnected")
        }
        
        micCurrentlyConnected = false
        onMicStateUpdate()
    }
    
    func onMicStateUpdate() {
        if micInitiallyFound && micCurrentlyConnected {
            self.micView.image = UIImage(systemName: "microphone.fill")
//            Task {
//                await liveSpeechRecognizer?.finishTranscribing()
//            }
            
            // make sure mode isn't narration
            if labelTypeSegmentedControl.selectedSegmentIndex == 1 {
                labelTypeSegmentedControl.selectedSegmentIndex = 0
                labelTypeSegmentedControlUpdated()
            }
            labelTypeSegmentedControl.setEnabled(false, forSegmentAt: 1)
        } else {
            self.micView.image = UIImage(systemName: "microphone.slash.fill")
            labelTypeSegmentedControl.setEnabled(true, forSegmentAt: 1)
            
//            if micInitiallyFound {
//                liveSpeechRecognizer?.startTranscribing()
//            }
        }
    }
    
    func computeARKitTimeOffset() {
        // set the ARKit time offset by converting from seconds from boot time to UNIX time
        let uptime = ProcessInfo.processInfo.systemUptime; // Get NSTimeInterval of uptime i.e. the delta: now - bootTime
        let nowTimeIntervalSince1970 = Date().timeIntervalSince1970
        arKitTimeOffset = nowTimeIntervalSince1970 - uptime;
    }
    
    func reconnectGoPro() {
        let defaults = UserDefaults.standard
        var goProName = defaults.object(forKey: "recordingGoPro") as? String
        if let goProName = goProName {
            print("attempting reconnection to GoPro")
            goProManager.attemptConnection(goProName: goProName, foundHandler: setGoPro)
        }
    }
    
    func setGoPro(_ gopro: GoPro) {
        let defaults = UserDefaults.standard
        defaults.set(gopro.name, forKey: "recordingGoPro")
        self.gopro = gopro
        
        if !useGoPro {
            disconnectGoPro()
        }
    }
    
    func disconnectGoPro() {
        gopro?.disconnect()
        goProManager.stopScanning()
        gopro = nil
    }
    
    func initializeSession() {
        // global world anchor
        worldAnchorCountdown = worldAnchorInitialCountdown
        justAddedWorldOrigin = false
        if worldAnchor != nil {
            session.remove(anchor: worldAnchor!)
            print("removed world anchor")
            worldAnchor = nil
        }
        
        // local world anchor
        localAnchorCountdown = localAnchorInitialCountdown
        if localWorldAnchor != nil {
            session.remove(anchor: localWorldAnchor!)
            print("removed local world anchor")
            localWorldAnchor = nil
        }
        
        // set recording icon to none
        
        
        // other
        isHostSide = true
        streamingPeerSessionId = nil
        demonstrationData = nil
        setRecordingState(false)
        
//        resetSocket()
        setRecordingMode(mode: .none)
    }
    
    func setRecordingControlsState(isRecording: Bool) {
        useGoProSwitch.isEnabled = !isRecording
        useViewerSwitch.isEnabled = !isRecording
        streamPeerSocketSwitch.isEnabled = !isRecording
        leftRightSegmentedControl.isEnabled = !isRecording
        recordTypeSegmentedControl.isEnabled = !isRecording
        labelTypeSegmentedControl.isEnabled = !isRecording
        demosButton.isEnabled = !isRecording
        goProButton.isEnabled = !isRecording
        resetPoseButton.isEnabled = !isRecording
        nameSessionButton.isEnabled = !isRecording
        noteButton.isEnabled = !isRecording
        
        updateTaskUI()
    }
    
    func setRecordingMode(mode: RecordingMode) {
        self.recordingMode = mode
        DispatchQueue.main.async {
            var shouldEnable = false
            
            switch mode {
            case .none:
                self.recordingModeIcon.image = UIImage(systemName: "xmark")
                shouldEnable = false
            case .single:
                self.recordingModeIcon.image = UIImage(systemName: "person.fill")
                shouldEnable = true
                self.leftRightSegmentedControl.isEnabled = !self.isRecording
            case .both:
                self.recordingModeIcon.image = UIImage(systemName: "person.2.fill")
                shouldEnable = true
                self.leftRightSegmentedControl.isEnabled = false
            }
            
            if self.gopro == nil {
                self.goProConnectedIcon.image = UIImage(systemName: "xmark")
            } else {
                self.goProConnectedIcon.image = UIImage(systemName: "camera.fill")
            }
            
            let goProConditionSatisfied = self.useGoPro ? self.gopro != nil : true
            let defaults = UserDefaults.standard
            var qrCalibrationRunName = (defaults.object(forKey: "qrCalibrationRunName") as? String)!
            var gripperCalibrationRunName = (defaults.object(forKey: "gripperCalibrationRunName") as? String)!
            let calibrationsSatisfied = self.getDemonstrationType() == .Demonstration ? ((!qrCalibrationRunName.isEmpty || !self.useGoPro) && !gripperCalibrationRunName.isEmpty): true // if not using GoPro then don't require QR calibration
            let tasksSpecifiedSatisfied = self.getDemonstrationType() == .Demonstration && self.tasksState.labelType == .Predefined ? self.tasksState.tasks.count > 0 : true
            let gripperWidthTaskID = (UserDefaults.standard.object(forKey: "gripperWidthTaskID") as? String)!
            let gripperWidthTaskIDSatisfied = self.getDemonstrationType() == .Demonstration && self.tasksState.labelType == .GripperWidth ? gripperWidthTaskID != "" : true
            let fpsConditionSatisfied = self.fps > 59 // sometimes ARKit drops to 30fps when too hot, so don't let the user record
            
            self.recordButton.isEnabled = (shouldEnable && goProConditionSatisfied && calibrationsSatisfied && tasksSpecifiedSatisfied && gripperWidthTaskIDSatisfied && fpsConditionSatisfied) || self.isRecording
        }
    }
    
    func session(_ session: ARSession, didOutputAudioSampleBuffer audioSampleBuffer: CMSampleBuffer) {
        // Process the audio sample buffer here
        if isRecording && micCurrentlyConnected && micInitiallyFound {
            demonstrationData?.logAudio(audioSampleBuffer: audioSampleBuffer)
        }
    }
        
    func addLocalWorldAnchor() {
        localWorldAnchor = ARAnchor(name: "local world frame", transform: matrix_identity_float4x4)
        session.add(anchor: localWorldAnchor!)
        print("Added local world frame")
        
        // Start looking for other players via MultiPeerConnectivity.
        multipeerSession = MultipeerSession(receivedDataHandler: receivedData, peerJoinedHandler:
                                            peerJoined, peerLeftHandler: peerLeft, peerDiscoveredHandler: peerDiscovered)
    }
    
    func addWorldAnchor() {
        worldAnchor = ARAnchor(name: "world frame", transform: matrix_identity_float4x4)
        session.add(anchor: worldAnchor!)
        print("Added world anchor for peer")
    }
    
    @IBAction func viewDemosButtonPress(_ sender: Any) {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        let secondVC = storyboard.instantiateViewController(identifier: "DemonstrationsViewController")

        secondVC.modalPresentationStyle = .fullScreen
        secondVC.modalTransitionStyle = .crossDissolve

        present(secondVC, animated: true, completion: nil)
//        session.pause()
    }
    
    @IBAction func viewTasksButtonPress(_ sender: Any) {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        let secondVC = storyboard.instantiateViewController(identifier: "TasksViewController") as TasksViewController

        secondVC.modalPresentationStyle = .fullScreen
        secondVC.modalTransitionStyle = .crossDissolve
        secondVC.onExit = { [weak self] in
            self?.updateTaskUI()
        }

        present(secondVC, animated: true, completion: nil)
    }
    
    enum MyError: Error {
        case error
    }
    
    @IBAction func connectGoProButtonPress(_ sender: Any) {
        goProManager.stopScanning()
        
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        let secondVC = storyboard.instantiateViewController(identifier: "GoProsViewController") as GoProsViewController

        secondVC.modalPresentationStyle = .fullScreen
        secondVC.modalTransitionStyle = .crossDissolve
        secondVC.delegate = self

        present(secondVC, animated: true, completion: nil)
    }
    
    func shouldRecordButtonBeNextTaskButton() -> Bool {
        // recording button should become next task button if we are on all tasks except the last task or if we are on the last task, but haven't confirmed it yet
        return isRecording && getDemonstrationType() == .Demonstration && tasksState.labelType == .Predefined && (tasksState.currentTaskIndex != tasksState.tasks.count - 1 || !tasksState.currentlyRecordingTask)
    }
    
    @IBAction func recordButtonPress(_ sender: UIButton) {
        recordButtonPress()
    }
    
    func recordButtonPress() {
        if !recordButton.isEnabled {
            return
        }
        
        if shouldRecordButtonBeNextTaskButton() {
            nextTaskButtonPress()
        } else {
            startStopRecording()
        }
    }
    
    func startStopRecording() {
        if !isRecording {
            // about to start recording
            recordingStartTime = Date()
            recordingName = DateManager.getISOFormatter().string(from: recordingStartTime!)
            
            var number = String()
            for _ in 1...5 {
               number += "\(Int.random(in: 1...9))"
            }
            
            recordingName += "_\(number)"
        }
        
        setRecordingState(!isRecording)
        
        if multipeerSession != nil {
            let multipeerSession = multipeerSession!
            if !multipeerSession.connectedPeers.isEmpty {
                let message = isRecording ? "start" : "stop"
//
//                let commandData = "Recording:\(message)".data(using: .utf8)!
//                multipeerSession.sendToAllPeers(commandData, reliably: true)
                                
                let data = RecordingMessage(startRecording: isRecording, recordingName: isRecording ? NSString(string:recordingName) : nil, recordingStartTime: isRecording ? recordingStartTime as NSDate? : nil)
                
                guard let encodedData = try? NSKeyedArchiver.archivedData(withRootObject: data, requiringSecureCoding: true)
                else { fatalError("Unexpectedly failed to encode recording message.") }
                // Use reliable mode if the data is critical, and unreliable mode if the data is optional.
                multipeerSession.sendToAllPeers(encodedData, reliably: true)
                
                print("sent recording message")
                print("Sent recording command to peer: \(message)")
            }
        }
    }
    
    func fpsDropAlert() {
        // Present message saying to restart app
        let alertController = UIAlertController(title: "Error", message: "Current recording was abandoned! This was due to FPS dropping below 60 likely due to overheating! Let the phone cool down and then restart the app to try again.", preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
            alertController.dismiss(animated: true, completion: nil)
            // do nothing
        }
        alertController.addAction(okAction)
        self.present(alertController, animated: true, completion: nil)
    }
    
    func abandonCurrentRecording() {
        if isRecording {
            print("Abandoning in progress recording!")
            demonstrationData = nil // setting to nil will prevent it from being saved
            startStopRecording()
        }
    }
    
    func getDemonstrationType() -> DemonstrationType {
        switch recordTypeSegmentedControl.selectedSegmentIndex {
        case 0: return .QRCalibration
        case 1: return .GripperCalibration
        case 2: return .Demonstration
        default:
            return .Demonstration
        }
    }
    
    func setRecordingState(_ newIsRecording: Bool) {
        // if newIsRecording, expects that `recordingName` has already been set
        
        isRecording = newIsRecording
        if isRecording {
            // recording started
            // recompute ARkit time offset in case it got messed up
            computeARKitTimeOffset()
            
            // determine the recording name
            // name for recording
            let side = isRightSide ? "right" : "left"
            var recordingName = "\(recordingName)".replacingOccurrences(of: ":", with: "-")
            
            // add session name to recording name
            let defaults = UserDefaults.standard
            var sessionName = (defaults.object(forKey: "sessionName") as? String)!
            if sessionName.isEmpty {
                sessionName = "no-session"
            }
            recordingName += "_\(sessionName)"
            
            // add demonstration type to recording name
            if getDemonstrationType() == .QRCalibration {
                recordingName += "_qrcalibration"
            } else if getDemonstrationType() == .GripperCalibration {
                recordingName += "_grippercalibration"
            } else {
                recordingName += "_demonstration"
            }
            
            // add side to recording name
            recordingName += "_\(side)"
            
            // create the demonstration data
            var qrLatencyMilliseconds = defaults.object(forKey: "timeIPhoneAheadOfQRinMS") as? Int
            if qrLatencyMilliseconds == nil {
                qrLatencyMilliseconds = 0
            }
            let qrCalibrationRunName = defaults.object(forKey: "qrCalibrationRunName") as? String
            let gripperCalibrationRunName = defaults.object(forKey: "gripperCalibrationRunName") as? String
            let note = defaults.object(forKey: "note") as? String
            let gripperWidthTaskID = (UserDefaults.standard.object(forKey: "gripperWidthTaskID") as? String)!

            let taskID = tasksState.labelType == .GripperWidth ? gripperWidthTaskID : nil
            
            demonstrationData = DemonstrationData(recordingName: recordingName, isRight: isRightSide, recordingStartTime: recordingStartTime!, timeIPhoneAheadOfQRinMS: qrLatencyMilliseconds!, demonstrationType: getDemonstrationType(), qrCalibrationRunName: qrCalibrationRunName!, gripperCalibrationRunName: gripperCalibrationRunName!, sessionName: sessionName, note: note!, hasGoPro: useGoPro, labelType: tasksState.labelType, taskID: taskID)
            
            // start GoPro recording
            if useGoPro {
                gopro!.startRecording()
            }
            
            // start audio transcription
            if getDemonstrationType() == .Demonstration && demonstrationData!.labelType == .Narration {
                Task {
//                    await liveSpeechRecognizer?.finishTranscribing()
                    
                    // if not using contact mic, then start speech recognition
                    if !micCurrentlyConnected {
                        entireDemoSpeechRecognizer = SpeechRecognizer(shouldReportPartialResults: false, callback: nil)
                        entireDemoSpeechRecognizer!.startTranscribing()
                    }
                }
            }
            
            // if qr calibration then need to enable this flag to indicate the calibration is starting
            isStartLoggingQR = true
            
            tasksState.reset()
            
            setRecordingControlsState(isRecording: true)
            
            // if this is a predefined demonstration, then see if we should start right away or confirm first
            if getDemonstrationType() == .Demonstration && demonstrationData!.labelType == .Predefined {
                if tasksState.tasks[0] == "CONFIRM" {
                    tasksState.currentTaskIndex += 1
                } else {
                    markTaskStart(time: Date())
                }
            }
        } else {
            // recording ended
            // stop GoPro recording
            if useGoPro {
                gopro?.stopRecording()
            }
            
            if demonstrationData != nil {
                Task { // have to do this processing aysnc because we need to wait until transcription finishes if present
                    // stop audio recording
                    var speechResult: SFSpeechRecognitionResult?
                    var speechStartTime: Date?
                    if getDemonstrationType() == .Demonstration && demonstrationData!.labelType == .Narration {
                        if let entireDemoSpeechRecognizer = entireDemoSpeechRecognizer {
                            await entireDemoSpeechRecognizer.finishTranscribing()
                            if entireDemoSpeechRecognizer.transcriptionSuccessful {
                                speechResult = entireDemoSpeechRecognizer.speechRecognitionResult
                                speechStartTime = await entireDemoSpeechRecognizer.startTranscriptionTimestamp
                            } else {
                                print("Transcription not successful or no speech detected")
                            }
                        }
                        
//                        if !micCurrentlyConnected {
//                            liveSpeechRecognizer!.startTranscribing() // restart live transcription
//                        }
                    }
                    
                    let demonstrationData = demonstrationData!
                    
                    // if task is in progress, mark it as done
                    if demonstrationData.labelType == .Narration {
                        markTaskEnd(time: Date())
                    }
                    if tasksState.currentlyRecordingTask {
                        markTaskEnd(time: Date(), enableAutoStep: false)
                    }
                    
                    // save the recording to disk
                    do {
                        // set recording name
                        let recordingName = demonstrationData.recordingName
                        if getDemonstrationType() == .QRCalibration {
                            let defaults = UserDefaults.standard
                            defaults.set(recordingName, forKey: "qrCalibrationRunName")
                        } else if getDemonstrationType() == .GripperCalibration {
                            let defaults = UserDefaults.standard
                            defaults.set(recordingName, forKey: "gripperCalibrationRunName")
                        }
                        
                        // final QR latency for QR latency calibration
                        let defaults = UserDefaults.standard
                        var qrLatencyMilliseconds = defaults.object(forKey: "timeIPhoneAheadOfQRinMS") as? Int
                        if qrLatencyMilliseconds == nil {
                            qrLatencyMilliseconds = 0
                        }
                                            
                        demonstrationData.setFinalData(timeIPhoneAheadOfQRinMS: qrLatencyMilliseconds, speechRecognitionResult: speechResult, transcriptionStartTime: speechStartTime, taskSegmentationEvents: tasksState.taskSegmentationEvents)
                        try demonstrationData.saveLocally()
                        let message = "Saved demonstration data: \(recordingName)"
                        messageLabel.displayMessage(message)
                        defaults.set("", forKey: "note")
                    } catch {
                        let message = "Failed to save demonstration data: \(recordingName)"
                        messageLabel.displayMessage(message)
                    }
                    
                    self.demonstrationData = nil
                    self.setRecordingControlsState(isRecording: false)
                }
            } else {
                self.setRecordingControlsState(isRecording: false)
            }
        }
        updateTaskUI()
    }
    
    func updateTaskUI() {
        // tasks buttons
        tasksButton.isEnabled = !self.isRecording
        nextTaskButton.isEnabled = self.isRecording
        switch labelTypeSegmentedControl.selectedSegmentIndex {
        case 0:
            tasksState.labelType = .None
            tasksButton.isEnabled = false
            nextTaskButton.isEnabled = false
            break
        case 1:
            tasksState.labelType = .Narration
            tasksButton.isEnabled = false
            break
        case 2:
            tasksState.labelType = .Predefined
            break
        case 3:
            tasksState.labelType = .GripperWidth
            tasksButton.isEnabled = false
            nextTaskButton.isEnabled = false
        default:
            // invalid
            break
        }
        
        // tasks from task list UI
        tasksState.tasks = (UserDefaults.standard.object(forKey: "tasks") as? [String])!
        
        // enabled segmented control
        recordTypeSegmentedControl.setEnabled(useGoPro, forSegmentAt: 0)
        if !useGoPro && recordTypeSegmentedControl.selectedSegmentIndex == -1 {
            // if timecode sync was selected and then useGoPro switch was disabled, then move to a different selection because timecode sync only makes sense for gopro
            recordTypeSegmentedControl.selectedSegmentIndex = 2
        }
        labelTypeSegmentedControl.isEnabled = getDemonstrationType() == .Demonstration && !isRecording
        
        // text above record button
        if isRecording {
            if getDemonstrationType()  == .Demonstration {
                switch demonstrationData!.labelType {
                case .None:
                    currentTaskLabel.text = "Not collecting task labels"
                case .Narration:
                    currentTaskLabel.text = "Narration Mode"
                case .Predefined:
                    if tasksState.tasks.count == 0 {
                        currentTaskLabel.text = "No task specified"
                    } else {
                        let taskName = tasksState.tasks[tasksState.currentTaskIndex]
                        if tasksState.currentlyRecordingTask {
                            currentTaskLabel.text = "\(taskName) (Recording)"
                        } else {
                            currentTaskLabel.text = "\(taskName) (Confirm?)"
                        }
                    }
                case .GripperWidth:
                    let gripperWidthTaskID = (UserDefaults.standard.object(forKey: "gripperWidthTaskID") as? String)!
                    currentTaskLabel.text = "Gripper Width Mode - \"\(gripperWidthTaskID)\""
                }
            } else {
                currentTaskLabel.text = ""
            }
        } else {
            if getDemonstrationType() == .Demonstration {
                switch tasksState.labelType {
                case .None:
                    currentTaskLabel.text = "Not collecting task labels"
                case .Narration:
                    currentTaskLabel.text = "Narration Mode"
                case .Predefined:
                    // recording not started, so list all the tasks
                    if tasksState.tasks.count == 0 {
                        currentTaskLabel.text = "No tasks specified"
                    } else {
                        currentTaskLabel.text = tasksState.tasks.joined(separator: " -> ")
                    }
                case .GripperWidth:
                    let gripperWidthTaskID = (UserDefaults.standard.object(forKey: "gripperWidthTaskID") as? String)!
                    if gripperWidthTaskID == "" {
                        currentTaskLabel.text = "Gripper Width Mode - Please specify task ID"
                    } else {
                        currentTaskLabel.text = "Gripper Width Mode - \"\(gripperWidthTaskID)\""
                    }
                }
            } else {
                currentTaskLabel.text = ""
            }
        }
        
        // recording button state
        if isRecording {
            if shouldRecordButtonBeNextTaskButton() {
                // recording button should become next task button if we are on all tasks except the last task or if we are on the last task, but haven't confirmed it yet
                self.recordButton.setTitle("Next Task", for: .normal)
                self.recordButton.setTitleColor(.white, for: .normal)
                self.recordButton.backgroundColor = .blue
            } else {
                // recording button should be stop button
                self.recordButton.setTitle("Stop Recording", for: .normal)
                self.recordButton.setTitleColor(.white, for: .normal)
                self.recordButton.backgroundColor = .red
            }
        } else {
            self.recordButton.setTitle("Start Recording", for: .normal)
            self.recordButton.setTitleColor(.systemBlue, for: .normal)
            self.recordButton.backgroundColor = .systemGray6
        }
    }
    
    @IBAction func recordTypeSegmentedControlValueChanged(_ sender: Any) {
        updateTaskUI()
    }
    
    @IBAction func leftRightSegmentedControlValueChanged(_ sender: Any) {
        isRightSide = leftRightSegmentedControl.selectedSegmentIndex == 1
        let defaults = UserDefaults.standard
        defaults.set(isRightSide, forKey: "isRight")
    }
    
    @IBAction func streamPeerToggleChange(_ sender: Any) {
        streamPeerSocket = streamPeerSocketSwitch.isOn
    }
    
    @IBAction func useGoProToggleChange(_ sender: Any) {
        setUseGoPro(isOn: useGoProSwitch.isOn)
    }
    
    func setUseGoPro(isOn: Bool) {
        useGoPro = isOn
        let defaults = UserDefaults.standard
        defaults.set(useGoPro, forKey: "useGoPro")
        
        if useGoPro {
            reconnectGoPro()
        } else {
            disconnectGoPro()
        }
        
        updateTaskUI()
    }
    
    @IBAction func nextTaskButtonPress(_ sender: Any) {
        nextTaskButtonPress()
    }
    
    func nextTaskButtonPress() {
        assert(isRecording)
        switch demonstrationData!.labelType {
        case .None:
            // stop the recording
            startStopRecording()
        case .Narration:
            // indicates end of the task
            markTaskEnd(time: Date())
        case .Predefined:
            if tasksState.currentlyRecordingTask {
                // case 1: we are currently recording a task so mark task as done
                markTaskEnd(time: Date())
            } else {
                // case 2: we are not currently recording a task, start the next task
                markTaskStart(time: Date())
            }
        case .GripperWidth:
            break
        }
    }
    
    @IBAction func useViewerToggleChange(_ sender: Any) {
        let defaults = UserDefaults.standard
        defaults.set(useViewerSwitch.isOn, forKey: "useViewer")
        
        // Present message saying to restart app
        let alertController = UIAlertController(title: "Warning", message: "Changes to viewer will only take effect after app is restarted", preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
            alertController.dismiss(animated: true, completion: nil)
            // do nothing
        }
        alertController.addAction(okAction)
        
        self.present(alertController, animated: true, completion: nil)
    }
    
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        var foundParticipantAnchor = false
        var foundWorldAnchor = false
        
        for anchor in anchors {
//            messageLabel.displayMessage("new anchor! \(anchor.name)", duration: 2.0)
//            let anchorEntity = AnchorEntity(anchor: anchor)
//            let coordinateSystem2 = MeshResource.generateCoordinateSystemAxes()
//            anchorEntity.addChild(coordinateSystem2)
//            arView.scene.addAnchor(anchorEntity)
            
            if let participantAnchor = anchor as? ARParticipantAnchor {
                if localWorldAnchor == nil {
                    print("found participantAnchor before we even added a localWorldAnchor, so just ignoring the call")
                    continue
                }
                
                foundParticipantAnchor = true
                print("found a new participantAnchor")
                if streamingPeerSessionId != nil {
//                    messageLabel.displayMessage("found participantAnchor, but streamingPeerSessionId is already set. Ignoring new participantAnchor, but this is probably a bad idea to do this.")
//                    continue
                    alertError(message: "streamingPeerSessionId already set, but found a new participantAnchor. streamingPeerSessionId: \(streamingPeerSessionId). participantAnchor: \(participantAnchor.sessionIdentifier?.uuidString)")
                }
                
                messageLabel.displayMessage("Established joint experience with a peer.")
                // ...
                let anchorEntity = AnchorEntity(anchor: participantAnchor)
                
                let coordinateSystem = MeshResource.generateCoordinateSystemAxes()
                anchorEntity.addChild(coordinateSystem)
                
                let color = participantAnchor.sessionIdentifier?.toRandomColor() ?? .white
                let coloredSphere = ModelEntity(mesh: MeshResource.generateSphere(radius: 0.03),
                                                materials: [SimpleMaterial(color: color, isMetallic: true)])
                anchorEntity.addChild(coloredSphere)
                
                arView?.scene.addAnchor(anchorEntity)
                
                //                if peerSessionIDs.count > 1 {
                //                    alertError(message: "More than 1 peer detected at once in session didAdd. Connected peers count: \(multipeerSession?.connectedPeers.count)")
                //                }
                
                streamingPeerSessionId = anchor.sessionIdentifier!.uuidString
                
                if streamingPeerSessionId! < session.identifier.uuidString {
                    isHostSide = true
                } else {
                    isHostSide = false
                }
                
                print("Now this phone has isHostSide: \(isHostSide)")
                
                
                
                
                // sanity check a bunch of things
                if localWorldAnchor == nil {
                    alertError(message: "there should be a local frame when participantAnchor added")
                }
                if worldAnchor != nil {
                    alertError(message: "there should not be a peer world frame when participantAnchor added")
                }
                if worldAnchorCountdown != worldAnchorInitialCountdown {
                    alertError(message: "world anchor countdown was not at its initial value of \(worldAnchorInitialCountdown). Instead was \(worldAnchorCountdown).")
                }
                
                // disable recording until we align world anchor from peer
                setRecordingMode(mode: .none)
            } else if anchor.name == "world frame" {
                foundWorldAnchor = true
                if streamingPeerSessionId == nil {
//                    alertError(message: "received world frame but streamingPeerSessionId is nil")
                    messageLabel.displayMessage("received world frame but streamingPeerSessionId is nil")
                    resetAll()
                    continue
                }
                
                // remaining
                let anchorEntity = AnchorEntity(anchor: anchor)
                let coordinateSystem = MeshResource.generateCoordinateSystemAxes()
                anchorEntity.addChild(coordinateSystem)
                
                arView?.scene.addAnchor(anchorEntity)
                
                if anchor.sessionIdentifier == session.identifier {
                    // world frame made my this phone
                    print("found world frame for current session")
                } else {
                    // world frame made by different phone
                    
//                    assert(anchor.sessionIdentifier?.uuidString == streamingPeerSessionId)
                    print("found world frame for other session")
                    
                    if isHostSide {
                        alertError(message: "did not expect to be the host and receive a world frame from peer")
                    }
                    
                    // update only non host to match host frame
                    if !isHostSide {
//                        var changePose = simd_inverse(anchor.transform)
//                        session.setWorldOrigin(relativeTransform: changePose)
//                        messageLabel.displayMessage("UPDATE THE WORLD POSE", duration: 10.0)
                        
//                        var pose = matrix_identity_float4x4;
//                        pose.columns.3.x = 10
//                        session.setWorldOrigin(relativeTransform: pose)
                        
//                        if anchor.sessionIdentifier?.uuidString != streamingPeerSessionId {
//                            alertError(message: "got a world frame from a different peer that isn't the streaming peer, which is unexpected. Expected: \(streamingPeerSessionId). Actual: \(anchor.sessionIdentifier!.uuidString)")
//                        }
                        
                        // for some reason anchor.sessionIdentifier doesn't seem to be set at all so we can't validate that it matches the streaming peer
                        
                        let pose = anchor.transform
                        session.setWorldOrigin(relativeTransform: pose)
                        messageLabel.displayMessage("Updated origin using world frame from peer")
                    }
                }
                
                justAddedWorldOrigin = true
                
                // remove local world anchor
                if localWorldAnchor != nil {
                    session.remove(anchor: localWorldAnchor!)
                    print("removed local world anchor")
                    localWorldAnchor = nil
                } else {
                    alertError(message: "expected there to be a local world frame at the point when the peer world frame was received")
                }
                
                // enable both sides recording mode now that we have the world frame
                setRecordingMode(mode: .both)
            } else if anchor.name == "local world frame" {
                if anchor.sessionIdentifier == session.identifier {
                    let anchorEntity = AnchorEntity(anchor: anchor)
                    let coordinateSystem = MeshResource.generateCoordinateSystemAxes()
                    anchorEntity.addChild(coordinateSystem)
                    
                    arView?.scene.addAnchor(anchorEntity)
                    
                    setRecordingMode(mode: .single)
                }
            }
        }
        
        if foundParticipantAnchor && foundWorldAnchor {
            alertError(message: "didAdd participant anchor and world anchor in the same update call")
        }
    }
    
    /// - Tag: DidOutputCollaborationData
    func session(_ session: ARSession, didOutputCollaborationData data: ARSession.CollaborationData) {
        guard let multipeerSession = multipeerSession else { return }
        if !multipeerSession.connectedPeers.isEmpty {
            guard let encodedData = try? NSKeyedArchiver.archivedData(withRootObject: data, requiringSecureCoding: true)
            else { fatalError("Unexpectedly failed to encode collaboration data.") }
            // Use reliable mode if the data is critical, and unreliable mode if the data is optional.
            let dataIsCritical = data.priority == .critical
            multipeerSession.sendToAllPeers(encodedData, reliably: dataIsCritical)
        } else {
//            print("Deferred sending collaboration to later because there are no peers.")
        }
    }
    
    func poseToText(_ pose: simd_float4x4, _ fps: Double) -> String {
        return "X: \(String(format: "%.3f", pose[3][0]))\nY: \(String(format: "%.3f", pose[3][1]))\nZ: \(String(format: "%.3f", pose[3][2]))\nFPS: \(String(format: "%.1f", fps))"
    }
    
    func alertError(message: String) {
        DispatchQueue.main.async {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "hh:mm:ss"
            let date = dateFormatter.string(from: Date())
            
            // Present the error that occurred.
            let alertController = UIAlertController(title: "Error: \(date)", message: message, preferredStyle: .alert)
            
            let restartAction = UIAlertAction(title: "Restart Session", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
                self.resetAll()
            }
            alertController.addAction(restartAction)
            let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
            }
            alertController.addAction(okAction)
            
            self.present(alertController, animated: true, completion: nil)
        }
    }
    
    // ARSessionDelegate method
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        setRecordingMode(mode: self.recordingMode) // refresh recording state
        
        // If there is one peer, then stream the pose of the peer in addition to this phone's pose
        var peerPoseText: String = ""
        let sideSocketName: String = isRightSide ? "Right" : "Left"
        var peerWorldText: String = ""
                
//        if peerSessionIDs.count > 1 {
//            alertError(message: "More than 1 peer detected at once. Connected peers: \(multipeerSession?.connectedPeers.count)")
//        }
                        
        if streamingPeerSessionId != nil {
            // find the corresponding frame that has the peer's pose (`sessionIdentifier` and `identifier` must match the peer's session ID
            for (index, anchor) in frame.anchors.enumerated() {
//                if anchor.name == "local world frame" {
//                    session.remove(anchor: anchor)
//                }
                
                if streamingPeerSessionId == anchor.sessionIdentifier?.uuidString && streamingPeerSessionId == anchor.identifier.uuidString { // the key assumption here is that the participant anchor has both the sessionIdentifier and the anchor identifier set to the session ID, which is a weird assumption and should probably be corrected later since there may be more anchors that have these conditions held
                    let transform = anchor.transform
                    let timestamp = frame.timestamp
                    let x = transform.columns.3.x
                    let y = transform.columns.3.y
                    let z = transform.columns.3.z
                    let fps = 1/(timestamp - self.prevTimestampThisDevice)
                    
                    let displayString = "Other iPhone:   x: \(String(format: "%.4f", transform[3][0])), y: \(String(format: "%.4f", transform[3][1])), z: \(String(format: "%.4f", transform[3][2])), fps: \(String(format: "%.3f", fps))"
//                    print(displayString)
                    prevTimestampOtherDevice = timestamp
                    if publishPose && streamPeerSocket {
                        let dataPacket = DataPacket(transformMatrix: transform, timestamp: timestamp)
                        
                        if socketClient.isConnected() {
                            socketClient.sendData(dataPacket.toBytes().base64EncodedString(), channel: "update\(sideSocketName)")
                        }
                    }
                    peerPoseText = poseToText(transform, fps)
                    
                    // compute which phone is left vs right based on the relative positions between the phones now that they share the same coordinate frame
                    if justAddedWorldOrigin {
                        justAddedWorldOrigin = false
                        var hostPose: simd_float4x4 = matrix_identity_float4x4
                        var otherPose: simd_float4x4 = matrix_identity_float4x4
                        if isHostSide {
                            hostPose = frame.camera.transform
                            otherPose = transform
                        } else {
                            hostPose = transform
                            otherPose = frame.camera.transform
                        }
                        
                        let hostToOtherPose = matrix_multiply(simd_inverse(hostPose), otherPose)
                        if hostToOtherPose.columns.3.x > 0 {
                            isRightSide = !isHostSide
                        } else {
                            isRightSide = isHostSide
                        }
                        
                        leftRightSegmentedControl.selectedSegmentIndex = isRightSide ? 1 : 0
                        let defaults = UserDefaults.standard
                        defaults.set(isRightSide, forKey: "isRight")
                        
                        messageLabel.displayMessage("set isRightSide=\(isRightSide)")
                    }
                    
                    break
                }
            }
            
            // find the world frame
//            if peerWorldAnchor != nil {
//                let peerWorldAnchorTransform = peerWorldAnchor!.transform
//                peerWorldText = "\n\nPeer World\nX: \(String(format: "%.4f", peerWorldAnchorTransform[3][0]))\nY: \(String(format: "%.4f", peerWorldAnchorTransform[3][1]))\nZ: \(String(format: "%.4f", peerWorldAnchorTransform[3][2]))"
//            }
            
            
            
//            for (index, anchor) in frame.anchors.enumerated() {
//                let identifierMatchCur = session.identifier.uuidString == anchor.identifier.uuidString
//                let identifierMatchOther = otherSessionId == anchor.identifier.uuidString
//                let sessionIdMatchCur = session.identifier.uuidString == anchor.sessionIdentifier?.uuidString
//                let sessionIdMatchOther = otherSessionId == anchor.sessionIdentifier?.uuidString
//                let transform = anchor.transform
//
//                print("Index\(index) Name: \(anchor.name) identifierMatchCur: \(identifierMatchCur) identifierMatchOther: \(identifierMatchOther) sessionIdMatchCur: \(sessionIdMatchCur) sessionIdMatchOther: \(sessionIdMatchOther) x: \(String(format: "%.4f", transform[3][0])), y: \(String(format: "%.4f", transform[3][1])), z: \(String(format: "%.4f", transform[3][2]))")
//            }
            
//            if !isRightHand {
//                for anchor in frame.anchors {
//                    if anchor.sessionIdentifier?.uuidString == otherSessionId && anchor.name == "world frame" {
//                        let newPose = anchor.transform
//                        session.setWorldOrigin(relativeTransform: newPose)
//                        print("aligning world")
//                        print(newPose)
//                        messageLabel.displayMessage("ALIGNED WORLD.", duration: 5.0)
//                    }
//                }
//            }
            
        }
        
        // Timecode correction
        let defaults = UserDefaults.standard
        var timeLatencyCorrectionMS = (defaults.object(forKey: "timeIPhoneAheadOfQRinMS") as? Int)!
        var timeQRCode = defaults.object(forKey: "timeQRCode") as? String ?? "None"
        let hasScannedQR = timeQRCode != "None"
        if hasScannedQR {
            let newFormatter = DateManager.getISOFormatter()
            let date = newFormatter.date(from: timeQRCode)
            if let date = date {
                timeQRCode = formatDateForDisplay(date)
            }
        }
        var currentSystemAdjustedTime = ""
        var actualSystemTime = formatDateForDisplay(Date())
        
        var currentDate = Date()
        var adjustedDate = currentDate.addingTimeInterval(TimeInterval(Float(-timeLatencyCorrectionMS)/1000))
        currentSystemAdjustedTime = formatDateForDisplay(adjustedDate)
        
        // stream pose of this iPhone
        let transform = frame.camera.transform
        let timestamp = frame.timestamp // in seconds
        let arPoseDate = arTimeStampToDate(timestamp)
        let x = transform.columns.3.x
        let y = transform.columns.3.y
        let z = transform.columns.3.z
        fps = 1 / (timestamp - self.prevTimestampThisDevice)
        
        if isRecording {
            // abort demonstration if fps drops from 60 to 30
            // sometimes the FPS temporarily drops right at the start of the recording and then goes back up to 60. In those cases we will only abandon the recording if the FPS drops after the first second of recording
//            if fps < 31 && currentDate.timeIntervalSince1970 - recordingStartTime!.timeIntervalSince1970 > 1 {
//                abandonCurrentRecording()
//                fpsDropAlert()
//            }
            
            let ultrawideImage = frame.getCapturedUltraWideImage()
            let ultrawideTime = frame.getUltraWideTimestamp()
            let ultrawideDate = ultrawideTime != nil ? arTimeStampToDate(ultrawideTime!) : nil
                        
            demonstrationData?.logFrame(pose: transform, poseTime: arPoseDate, rgb: frame.capturedImage, depthMap: frame.sceneDepth?.depthMap, depthConfidenceMap: frame.sceneDepth?.confidenceMap, ultrawidergb: ultrawideImage, arkitTimestamp: frame.timestamp)
        }

        let displayString = "Cur iPhone:     x: \(String(format: "%.4f", transform[3][0])), y: \(String(format: "%.4f", transform[3][1])), z: \(String(format: "%.4f", transform[3][2])), fps: \(String(format: "%.3f", fps))"
//        print(displayString)
        prevTimestampThisDevice = timestamp
        if publishPose {
            let dataPacket = DataPacket(transformMatrix: transform, timestamp: timestamp)
            if socketClient.isConnected() {
                socketClient.sendData(dataPacket.toBytes().base64EncodedString(), channel: "update\(sideSocketName)")
            }
        }
        
        // update the UI label
        var minePoseText = poseToText(transform, fps)
        poseLabel.text = ""
        if peerPoseText == "" {
            poseLabel.text! += minePoseText
        } else {
            poseLabel.text! += "Mine\n" + minePoseText + "\n\nPeer\n" + peerPoseText
        }
        
        // find the world frame
//        let mineWorldAnchorTransform = worldAnchor!.transform
//        var mineWorldText = "Mine World\nX: \(String(format: "%.4f", mineWorldAnchorTransform[3][0]))\nY: \(String(format: "%.4f", mineWorldAnchorTransform[3][1]))\nZ: \(String(format: "%.4f", mineWorldAnchorTransform[3][2]))"
//        worldLabel.text = "\(mineWorldText)\n\n\(peerWorldText)"
        
        // update the status label
        let sessionName = (defaults.object(forKey: "sessionName") as? String)!
        let note = (defaults.object(forKey: "note") as? String)!
        
        stateLabel.text = ""
        if useGoPro {
            stateLabel.text! += "Last Time Sync: \(timeQRCode)\n" +
                                "System Time:     \(actualSystemTime)\n" +
                                "Adj Sys Time:     \(currentSystemAdjustedTime)\n" +
                                "iPhone ahead of QR by \(timeLatencyCorrectionMS)ms\n"
        }
        
        stateLabel.text! += "Socket: \(socketClient.getStatus())\n" +
                           "Session: \(sessionName)\n" +
                           "Note: \(note)"
        
        if useGoPro {
            if let gopro = gopro {
                stateLabel.text! += "\nGoPro: \(gopro.name) (connected)"
            } else {
                var savedGoProName = defaults.object(forKey: "recordingGoPro") as? String
                if let savedGoProName = savedGoProName {
                    if useGoPro {
                        stateLabel.text! += "\nGoPro: \(savedGoProName) (connecting...)"
                    }
                } else {
                    stateLabel.text! += "\nGoPro: Please pair"
                }
            }
        }
        
        // read QR code from the image captured by the iPhone
        if (getDemonstrationType() == .QRCalibration && isRecording) {
            if readQRInterval == 0 {
                readQRInterval = Constants.readQRIntervalStartValue
                let image = CIImage(cvPixelBuffer: frame.capturedImage)
                let qrStrings = parseQR(image: image)
                if (qrStrings.count > 0) {
                    let qrStringISO8601 = qrStrings[0]
                    let newFormatter = DateManager.getISOFormatter()
                                        
                    // if the QR code actually contains a timecode sync
                    if let qrCodeTime = newFormatter.date(from: qrStringISO8601) {
                        // compare QR date to AR date and system time
                        let QRvsARdifferenceInMS = Int(arPoseDate.timeIntervalSince(qrCodeTime)*1000)
    //                    print("found a QR code with value: \(qrString)")
                        
//                        let systemTime = Date()
    //                        let QRvsSystemdifferenceInMS = Int(systemTime.timeIntervalSince(qrCodeTime)*1000)
    //                    print("System time: \(QRvsSystemdifferenceInMS)ms after QR    AR timestamp is: \(QRvsARdifferenceInMS)ms after QR")
                        
                        if isStartLoggingQR {
                            // override QR latency right after the toggle is pressed
                            defaults.set(QRvsARdifferenceInMS, forKey: "timeIPhoneAheadOfQRinMS")
                            isStartLoggingQR = false
                        } else {
                            // update the QR latency with exponential weighted average
                            var oldCorrection = defaults.object(forKey: "timeIPhoneAheadOfQRinMS") as? Int
                            let alpha = 0.95
                            var newCorrection = Int(alpha * Double(oldCorrection!) + (1 - alpha) * Double(QRvsARdifferenceInMS))
                            defaults.set(newCorrection, forKey: "timeIPhoneAheadOfQRinMS")
                            defaults.set(qrStringISO8601, forKey: "timeQRCode")
                        }
                    }
                }
            } else {
                readQRInterval -= 1
            }
            
        }
                
        // temporary: print AR time and actual time
//        let arDate = arTimeStampToDate(timestamp)
//        let systemDate = Date()
//        print("AR date: \(arDate.timeIntervalSince1970)    actual date: \(systemDate.timeIntervalSince1970)")
        
        // countdown to add world anchor
        if streamingPeerSessionId != nil {
            if worldAnchorCountdown == 0 {
                if isHostSide {
                    addWorldAnchor()
                }
                worldAnchorCountdown -= 1
            } else if worldAnchorCountdown > 0 {
                worldAnchorCountdown -= 1
            }
        }
        
        // countdown to add local anchor
        if localAnchorCountdown == 0 {
            addLocalWorldAnchor()
            localAnchorCountdown -= 1
        } else if localAnchorCountdown > 0 {
            localAnchorCountdown -= 1
        }
    }
    
    func markTaskStart(time: Date) {
        assert(!tasksState.currentlyRecordingTask)
        tasksState.currentlyRecordingTask = true
        let taskSegmentationEvent = TaskSegmentation()
        taskSegmentationEvent.taskStart = time
        
        if demonstrationData!.labelType == .Predefined {
            // if repeating predefined tasks, we already know the sequence of tasks so we can assign the language label now
            taskSegmentationEvent.name = tasksState.tasks[tasksState.currentTaskIndex]
        }
        
        tasksState.taskSegmentationEvents.append(taskSegmentationEvent)
        
        updateTaskUI()
    }
    
    func markTaskEnd(time: Date, newTaskStartTime: Date? = nil, enableAutoStep: Bool = true) {
        switch demonstrationData!.labelType {
        case .Narration:
            assert(!tasksState.currentlyRecordingTask) // only mark the end with narration
            
            // create segmentation event because there was no explicit start
            let taskSegmentationEvent = TaskSegmentation()
            tasksState.taskSegmentationEvents.append(taskSegmentationEvent)
            
        case .Predefined:
            assert(tasksState.currentlyRecordingTask) // task should have been previously marked as started
        case .None:
            assert(false)
        case .GripperWidth:
            assert(false)
        }
        
        tasksState.currentlyRecordingTask = false
        tasksState.taskSegmentationEvents[tasksState.taskSegmentationEvents.count - 1].taskEnd = time
        
        if demonstrationData!.labelType == .Predefined && enableAutoStep {
            if tasksState.currentTaskIndex == tasksState.tasks.count - 1 {
                // end the demonstration
                startStopRecording()
            } else {
                tasksState.currentTaskIndex += 1
                
                // if the next task is a confirmation, then wait until starting the next task
                if tasksState.tasks[tasksState.currentTaskIndex] == "CONFIRM" {
                    // don't record data for the confirmation task, but also don't start the next task yet
                    tasksState.currentTaskIndex += 1
                } else {
                    var newTaskStartTime = newTaskStartTime
                    if newTaskStartTime == nil {
                        newTaskStartTime = time
                    }
                    // if no confirmation then just start the next task right away
                    markTaskStart(time: newTaskStartTime!)
                }
            }
        }
        
        updateTaskUI()
    }
    
    func narrationCallback(result: SFSpeechRecognitionResult, startIndex: Int) {
        result.bestTranscription.segments[startIndex..<result.bestTranscription.segments.count].forEach { segment in
            segment.substring.split(separator: " ").forEach {word in
                let word = String(word)
                if NarrationCommands.isDoneWord(word) {
                    handleLiveDoneWord()
                } else if NarrationCommands.isFinishedWord(word) {
                    handleLiveFinishWord()
                } else if NarrationCommands.isStartWord(word) {
                    handleLiveStartWord()
                }
            }
        }
    }
    
    func handleLiveStartWord() {
        if !isRecording {
            startStopRecording()
        } else {
            switch demonstrationData!.labelType {
            case .None:
                break
            case .Narration:
                break
            case .Predefined:
                if !tasksState.currentlyRecordingTask {
                    // we are not currently recording a task so start next task
                    markTaskStart(time: Date())
                }
            case .GripperWidth:
                break
            }
        }
    }
    
    func handleLiveDoneWord() {
        if isRecording {
            switch demonstrationData!.labelType {
            case .None:
                // stop the recording
                startStopRecording()
            case .Narration:
                // already handled in narration processing
                break
            case .Predefined:
                if tasksState.currentlyRecordingTask {
                    // we are currently recording a task so mark task as done
                    markTaskEnd(time: Date())
                }
            case .GripperWidth:
                // stop the recording
                startStopRecording()
            }
        }
    }
    
    func handleLiveFinishWord() {
        if isRecording {
            if getDemonstrationType() == .Demonstration && tasksState.labelType == .Predefined && !tasksState.currentlyRecordingTask {
                // if a task needs to be confirmed still, saying "finish" shouldn't do anything
                return
            }
            recordButtonPress() // mostly likely stop recording, but if in predefined mode this will move to next task
        }
    }
    
    func parseQR(image: CIImage) -> [String] {
        let detector = CIDetector(ofType: CIDetectorTypeQRCode,
                                  context: nil,
                                  options: [CIDetectorAccuracy: CIDetectorAccuracyHigh])

        let features = detector?.features(in: image) ?? []

        return features.compactMap { feature in
            return (feature as? CIQRCodeFeature)?.messageString
        }
    }
    
    func arTimeStampToDate(_ arTimeStamp: TimeInterval) -> Date {
        // for debugging:
//        let uptime = ProcessInfo.processInfo.systemUptime;
//        let nowTimeIntervalSince1970 = Date().timeIntervalSince1970
//        print("uptime \(uptime) now \(nowTimeIntervalSince1970) offset \(arKitTimeOffset) arTimeStamp: \(arTimeStamp) corrected time \(arTimeStamp + arKitTimeOffset)")
        
        return Date(timeIntervalSince1970: arTimeStamp + arKitTimeOffset)
    }

    func receivedData(_ data: Data, from peer: MCPeerID) {
        // collaboration data
        if let collaborationData = try? NSKeyedUnarchiver.unarchivedObject(ofClass: ARSession.CollaborationData.self, from: data) {
            session.update(with: collaborationData)
            return
        }
        
        // session ID
        let sessionIDCommandString = "SessionID:"
        if let commandString = String(data: data, encoding: .utf8), commandString.starts(with: sessionIDCommandString) {
            let newSessionID = String(commandString[commandString.index(commandString.startIndex,
                                                                     offsetBy: sessionIDCommandString.count)...])
            // If this peer was using a different session ID before, remove all its associated anchors.
            // This will remove the old participant anchor and its geometry from the scene.
            if let oldSessionID = peerSessionIDs[peer] {
                removeAllAnchorsOriginatingFromARSessionWithID(oldSessionID)
                                
                if oldSessionID == streamingPeerSessionId {
                    streamingPeerSessionId = newSessionID
                    
                    alertError(message: "Maybe not an error, but received new session ID from peer so things might break")
                }
            }
            
            peerSessionIDs[peer] = newSessionID
//            streamingPeerSessionId = newSessionID
            
            return
        }
        
        // recording start/stop
        if let recordingMessage = try? NSKeyedUnarchiver.unarchivedObject(ofClasses: [RecordingMessage.self, NSString.self, NSDate.self], from: data) as? RecordingMessage {
            if recordingMessage.startRecording {
                recordingName = recordingMessage.recordingName! as String
                recordingStartTime = recordingMessage.recordingStartTime! as Date
                messageLabel.displayMessage("Received start recording command with name: \(recordingMessage.recordingName!)")
            } else {
                messageLabel.displayMessage("Received end recording command")
            }
            
            setRecordingState(recordingMessage.startRecording)
            return
        }
        
        messageLabel.displayMessage("Received end recording command")
    }
    
    func peerDiscovered(_ peer: MCPeerID) -> Bool {
        guard let multipeerSession = multipeerSession else { return false }
        
        if multipeerSession.connectedPeers.count > 1 {
            // Do not accept more than two users in the experience.
            messageLabel.displayMessage("A third peer wants to join the experience.\nThis app is limited to two users.")
            return false
        } else {
            return true
        }
    }
    /// - Tag: PeerJoined
    func peerJoined(_ peer: MCPeerID) {
        messageLabel.displayMessage("A peer wants to join the experience. Hold the phones next to each other.")
        // Provide your session ID to the new user so they can keep track of your anchors.
        sendARSessionIDTo(peers: [peer])
    }
        
    func peerLeft(_ peer: MCPeerID) {
        messageLabel.displayMessage("A peer has left the shared experience.")
        
        // Remove all ARAnchors associated with the peer that just left the experience.
        if let sessionID = peerSessionIDs[peer] {
            if sessionID == streamingPeerSessionId {
                messageLabel.displayMessage("The STREAMING peer has left the shared experience.")
                streamingPeerSessionId = nil
                resetAll()
            }
            
            removeAllAnchorsOriginatingFromARSessionWithID(sessionID)
            peerSessionIDs.removeValue(forKey: peer)
            // TODO maybe something needs to be done here like only resetAll if the peer that left was actually the one we were streaming to streamingPeerId
        }
    }
    
    @IBAction func nameSessionButtonPress(_ sender: Any) {
        //1. Create the alert controller.
        let alert = UIAlertController(title: "Enter session name", message: "Saved as part of the demonstration to label an entire set of collected demonstrations. This will persist for each demonstration until changed.", preferredStyle: .alert)

        //2. Add the text field. You can configure it however you need.
        alert.addTextField { (textField) in
            let defaults = UserDefaults.standard
            textField.text = (defaults.object(forKey: "sessionName") as? String)!
        }

        // 3. Grab the value from the text field, and print it when the user clicks OK.
        alert.addAction(UIAlertAction(title: "Cancel", style: .default, handler: { [weak alert] (_) in
            alert?.dismiss(animated: true)
        }))
        alert.addAction(UIAlertAction(title: "Set", style: .default, handler: { [weak alert] (_) in
            let textField = alert!.textFields![0] // Force unwrapping because we know it exists.
//            print("Text field: \(textField.text)")
            let defaults = UserDefaults.standard
            defaults.set(textField.text!.replacingOccurrences(of: "_", with: "-"), forKey: "sessionName")
        }))
        
        // 4. Present the alert.
        self.present(alert, animated: true, completion: nil)
    }
    
    @IBAction func labelTypeSegmentedControlValueUpdate(_ sender: Any) {
        labelTypeSegmentedControlUpdated()
    }
    
    func labelTypeSegmentedControlUpdated() {
        UserDefaults.standard.set(labelTypeSegmentedControl.selectedSegmentIndex, forKey: "labelSelectedSegmentID")
        
        updateTaskUI()
        
        // if gripper width mode, then prompt user to enter the demonstration ID
        if tasksState.labelType == .GripperWidth {
            let alert = UIAlertController(title: "Enter task ID", message: "Task segmentation by gripper width is done in post-processing. The task ID identifies the task so we can manually setup the width-based segmentation later.", preferredStyle: .alert)

            alert.addTextField { (textField) in
                let defaults = UserDefaults.standard
                textField.text = (defaults.object(forKey: "gripperWidthTaskID") as? String)!
            }

            alert.addAction(UIAlertAction(title: "Cancel", style: .default, handler: { [weak alert] (_) in
                alert?.dismiss(animated: true)
            }))
            alert.addAction(UIAlertAction(title: "Set", style: .default, handler: { [weak alert] (_) in
                let textField = alert!.textFields![0] // Force unwrapping because we know it exists.
    //            print("Text field: \(textField.text)")
                let defaults = UserDefaults.standard
                defaults.set(textField.text?.lowercased(), forKey: "gripperWidthTaskID")
                self.updateTaskUI()
            }))
            
            // 4. Present the alert.
            self.present(alert, animated: true, completion: nil)
        }
    }
    
    @IBAction func noteButtonPress(_ sender: Any) {
        //1. Create the alert controller.
        let alert = UIAlertController(title: "Enter note", message: "Notes are saved as part of the demonstration and are meant to record info about a single demonstration. The note is reset after a demonstration is collected.", preferredStyle: .alert)

        //2. Add the text field. You can configure it however you need.
        alert.addTextField { (textField) in
            let defaults = UserDefaults.standard
            textField.text = (defaults.object(forKey: "note") as? String)!
        }

        // 3. Grab the value from the text field, and print it when the user clicks OK.
        alert.addAction(UIAlertAction(title: "Cancel", style: .default, handler: { [weak alert] (_) in
            alert?.dismiss(animated: true)
        }))
        alert.addAction(UIAlertAction(title: "Set", style: .default, handler: { [weak alert] (_) in
            let textField = alert!.textFields![0] // Force unwrapping because we know it exists.
//            print("Text field: \(textField.text)")
            let defaults = UserDefaults.standard
            defaults.set(textField.text, forKey: "note")
        }))
        
        // 4. Present the alert.
        self.present(alert, animated: true, completion: nil)
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        guard error is ARError else { return }
        
        let errorWithInfo = error as NSError
        let messages = [
            errorWithInfo.localizedDescription,
            errorWithInfo.localizedFailureReason,
            errorWithInfo.localizedRecoverySuggestion
        ]
        
        // Remove optional error messages.
        let errorMessage = messages.compactMap({ $0 }).joined(separator: "\n")
        
        DispatchQueue.main.async {
            // Present the error that occurred.
            let alertController = UIAlertController(title: "The AR session failed.", message: errorMessage, preferredStyle: .alert)
            let restartAction = UIAlertAction(title: "Restart Session", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
                self.resetAll()
            }
            alertController.addAction(restartAction)
            self.present(alertController, animated: true, completion: nil)
        }
    }
    
    @IBAction func resetTrackingButtonPress(_ sender: Any) {
        resetAll()
    }
    
    func resetAll() {
        multipeerSession?.endSession()
        
        
        for (peerId, sessionId) in peerSessionIDs {
            removeAllAnchorsOriginatingFromARSessionWithID(sessionId)
            peerSessionIDs.removeValue(forKey: peerId)
        }
        
        initializeSession()
        
        print("Resetting tracking")
        session.run(configuration!, options: [.resetTracking, .removeExistingAnchors])
        
//        multipeerSession = MultipeerSession(receivedDataHandler: receivedData, peerJoinedHandler:
//                                            peerJoined, peerLeftHandler: peerLeft, peerDiscoveredHandler: peerDiscovered)
    }
    
    override var prefersStatusBarHidden: Bool {
        // Request that iOS hide the status bar to improve immersiveness of the AR experience.
        return true
    }
    
    override var prefersHomeIndicatorAutoHidden: Bool {
        // Request that iOS hide the home indicator to improve immersiveness of the AR experience.
        return true
    }
    
    private func removeAllAnchorsOriginatingFromARSessionWithID(_ identifier: String) {
        guard let frame = session.currentFrame else { return }
        for anchor in frame.anchors {
            guard let anchorSessionID = anchor.sessionIdentifier else { continue }
            if anchorSessionID.uuidString == identifier {
                session.remove(anchor: anchor)
            }
        }
    }
    
    private func sendARSessionIDTo(peers: [MCPeerID]) {
        guard let multipeerSession = multipeerSession else { return }
        let idString = session.identifier.uuidString
        let command = "SessionID:" + idString
        if let commandData = command.data(using: .utf8) {
            multipeerSession.sendToPeers(commandData, reliably: true, peers: peers)
        }
    }
    
    func goProFormatToDate(_ urlString: String) -> Date? {
        // example urlString: oT240828192816.794oTD1oTZ-8oTI62560
        // oTI indicates some machine code which we ignore
        // Extract the timestamp part from the string
        let timestampPart = String(urlString.dropFirst(2).prefix(15)) // "240828183808.284"

        // Define a date formatter
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyMMddHHmmss.SSS"
        dateFormatter.timeZone = TimeZone(secondsFromGMT: 0) // Default to GMT

        // Convert the timestamp to a Date object
        if let date = dateFormatter.date(from: timestampPart) {
            // Extract DST indicator part between "oTD" and "oTZ"
            if let tdStartRange = urlString.range(of: "oTD"),
               let tzStartRange = urlString.range(of: "oTZ"){
                let tdRange = tdStartRange.upperBound..<tzStartRange.lowerBound
                
                let dstIndicatorPart = String(urlString[tdRange])
                let isDaylightSaving = dstIndicatorPart == "1" // Assuming '1' indicates DST is observed

                // Extract the timezone offset part between "oTZ" and "oTI"
                if let tzEndRange = urlString.range(of: "oTI") {
                    let tzRange = tzStartRange.upperBound..<tzEndRange.lowerBound
                    
                    let timezoneOffsetPart = String(urlString[tzRange])
                    
                    // Convert the timezone offset part to an integer
                    if var timezoneOffset = Float(timezoneOffsetPart) {
                        if Int(timezoneOffset) % 15 == 0 {
                            // in this case timezone offset is in minutes not hours
                            timezoneOffset /= 60
                        }
                        
                        // Convert hours to seconds
                        var timezoneOffsetInSeconds = Int(timezoneOffset * 3600)
                        
                        // Adjust for daylight saving time if necessary
                         if isDaylightSaving {
                             timezoneOffsetInSeconds += 3600 // Adding 1 hour (3600 seconds) for DST
                         }
                        
                        let adjustedDate = date.addingTimeInterval(TimeInterval(-timezoneOffsetInSeconds))
                        
                        return adjustedDate
                    } else {
                        print("Failed to parse timezone offset")
                    }
                } else {
                    print("Failed to find timezone offset in the string")
                }
            } else {
                print("Failed to find DST indicator in the string")
            }
        } else {
            print("Failed to parse date")
        }
        return nil
    }
    
    // Function to convert a GMT date to a local timezone string
    func formatDateForDisplay(_ date: Date, _ includeMS: Bool = false) -> String {
        // 1. Define a DateFormatter to format the date
        let dateFormatter = DateFormatter()
        
        // Set the formatter's time zone to the local time zone
        dateFormatter.timeZone = TimeZone.current
        
        // Set the desired date format
        if includeMS {
            dateFormatter.dateFormat = "yyyy-MM-dd hh:mm:ss:SSS a" // Customize the format as needed
        } else {
            dateFormatter.dateFormat = "yyyy-MM-dd hh:mm:ss a" // Customize the format as needed
        }
        
        
        // Convert the date to a formatted string in the local timezone
        let localDateString = dateFormatter.string(from: date)
        
        return localDateString
    }
    
    // Helper function to get the current date in custom format
    func formatCurrentDateAsGoPro() -> String {
        // 1. Get the current date and time
        let currentDate = Date()
        print(currentDate)
        
        // 2. Define DateFormatter for the custom format
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyMMddHHmmss.SSS"
        
        // Convert the current date to the timestamp format
        let timestampPart = dateFormatter.string(from: currentDate)
        
        // 3. Determine DST status
        let calendar = Calendar.current
        let timeZone = TimeZone.current
        let isDaylightSaving = timeZone.isDaylightSavingTime(for: currentDate) ? "1" : "0"
        
        // 4. Get timezone offset in hours
        var offset = timeZone.secondsFromGMT(for: currentDate)
        
        if offset % 3600 == 0 {
            offset /= 3600 // seconds to hours
            
            if isDaylightSaving == "1" {
                offset -= 1 // 1 hour
            }
        } else {
            offset /= 60 // seconds to minutes
            
            if isDaylightSaving == "1" {
                offset -= 60 // 60 minutes
            }
        }
        
        let formattedOffset = String(abs(offset))
        let offsetSign = offset >= 0 ? "" : "-"
        
        // Combine sign with formatted offset
        let timezoneOffsetPart = "\(offsetSign)\(formattedOffset)"
        
        // 5. Create the final formatted string
        let formattedString = "oT\(timestampPart)oTD\(isDaylightSaving)oTZ\(timezoneOffsetPart)oTI00000"
        
        return formattedString
    }
    
    @IBAction func deployButtonPress(_ sender: Any) {
        let defaults = UserDefaults.standard
        defaults.set("deployment", forKey: "appMode")
        
        // Present message saying to restart app
        let alertController = UIAlertController(title: "Warning", message: "Close and reopen the app to enter deployment mode", preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
            alertController.dismiss(animated: true, completion: nil)
            // do nothing
        }
        alertController.addAction(okAction)
        self.present(alertController, animated: true, completion: nil)
    }
}
