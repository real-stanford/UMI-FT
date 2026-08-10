//
//  GoPro.swift
//  umi_day_iphone
//
//  Created by Austin Patel on 9/19/24.
//  Copyright © 2024 Apple. All rights reserved.
//

import Foundation

class GoPro {
    private var peripheral: Peripheral
    var cameraStatus: CameraStatus?
    var lastCommand: Data?
    
    init (_ peripheral: Peripheral) {
        self.peripheral = peripheral
    }
    
    var name: String {
        return self.peripheral.name
    }
    
    func startRecording() {
        sendCameraCommand(command: Data([0x01, 0x01, 0x01]))
    }
    
    func stopRecording() {
        sendCameraCommand(command: Data([0x01, 0x01, 0x00]))
    }
    
    func sleep() {
        sendCameraCommand(command: Data([0x05]), ignoreFailure: true) // ignore failure because camera will have disconnected due to sleep command so we don't expect a response
    }
    
    func disconnect() {
        peripheral.disconnect()
    }
    
    private func sendCameraCommand(command: Data, ignoreFailure: Bool = false) {
        self.lastCommand = command
        self.peripheral.setCommand(command: command) { result in
            switch result {
            case .success(let response):
                //Check command/response and do something
                let commandResponse: CommandResponse = response
                if ((self.lastCommand![0] == 0x01) && (commandResponse.response[1] == 0x01)){
                    //Shutter Command
                    if (self.lastCommand![2] == 0x01){
                        self.cameraStatus!.busy = true
                    } else {
                        self.cameraStatus!.busy = false
                    }
                } else if (commandResponse.response[1] == 0x17){
                    if (commandResponse.response[2] == 0x00){
                        //Enable WiFi Command
//                        self.requestWiFiSettngs()
                        print("why is wifi requested to be enabled")
                    }
                }
            case .failure(let error):
                if !ignoreFailure {
                    NSLog("\(error)")
                    self.getCameraStatus()
                }
            }
        }
    }
    
    private func getCameraStatus() {
        peripheral.requestCameraStatus() { result in
            switch result {
            case .success(let status):
                print("Camera Status: \(status)")
                self.cameraStatus = status
            case .failure(let error):
                print("\(error)")
                self.getCameraStatus()
            }
        }
    }
}
