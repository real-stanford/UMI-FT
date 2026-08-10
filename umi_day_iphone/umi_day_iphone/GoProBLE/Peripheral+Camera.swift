/*
WunderLINQ Client Application
Copyright (C) 2020  Keith Conger, Black Box Embedded, LLC
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

import Foundation
import CoreBluetooth

/// A struct representing the camera's Wi-Fi settings
struct WiFiSettings {
    var SSID: String
    let password: String
}

/// A struct representing the camera's status
struct CameraStatus {
    var busy: Bool
    var mode: UInt8
    var wifiEnabled: Bool
    var previewAvailable: Bool
    var openGoPro: Bool
}

struct CommandResponse {
    var command: Data
    var response: Data
}

extension Peripheral {
    
    /// Send Command to camera
    /// - Parameter completion: The completion handler with an optional error that is invoked once the request completes.
    ///
    func setCommand(command: Data, completion: ((Result<CommandResponse, Error>) -> Void)?) {
        let serviceUUID = CBUUID(string: "FEA6")
        let commandUUID = CBUUID(string: "B5F90072-AA8D-11E3-9046-0002A5D5C51B")
        let commandResponseUUID = CBUUID(string: "B5F90073-AA8D-11E3-9046-0002A5D5C51B")
        let data = Data([UInt8(command.count)] + command)

        let finishWithResult: (Result<CommandResponse, Error>) -> Void = { result in
            // make sure to dispatch the result on the main thread
            DispatchQueue.main.async {
                completion?(result)
            }
        }
        
        registerObserver(serviceUUID: serviceUUID, characteristicUUID: commandResponseUUID) { data in
            // The response to a command is expected to be 3 bytes
            if data.count != 3 {
                finishWithResult(.failure(CameraError.invalidResponse))
                return
            }

            // The third byte represents the camera response. If the byte is 0x00 then the request was successful
            if (data[1] != 0x02) {
                finishWithResult(.failure(CameraError.responseError))
                return
            }
            finishWithResult(.success(CommandResponse(command:command, response:data)))
        } completion: { [weak self] error in
            // Check that we successfully enable the notification for the response before writing to the characteristic
            if error != nil { finishWithResult(.failure(error!)); return }
            self?.write(data: data, serviceUUID: serviceUUID, characteristicUUID: commandUUID) { error in
                if error != nil { finishWithResult(.failure(error!)) }
            }
        }
    }

    /// Reads camera's Wi-Fi settings
    /// - Parameter completion: The completion handler with a result representing either a success or a failure.
    ///                         In the success case, the associated value is an instance of WiFiSettings
    ///
    /// Discussion:
    /// Requesting Wi-Fi setting consists in reading characteristics GP-0002 (SSID) and GP-0003 (password)
    /// on the GoPro WiFi Access Point service

    func requestWiFiSettings(_ completion: ((Result<WiFiSettings, Error>) -> Void)?) {
        let serviceUUID = CBUUID(string: "B5F90001-AA8D-11E3-9046-0002A5D5C51B")
        var SSID: String?
        var password: String?

        let finishWithResult: (Result<WiFiSettings, Error>) -> Void = { result in
            // make sure to dispatch the result on the main thread
            DispatchQueue.main.async {
                completion?(result)
            }
        }

        let reads = DispatchGroup()
        reads.enter()
        readData(from: CBUUID(string: "B5F90002-AA8D-11E3-9046-0002A5D5C51B"), serviceUUID: serviceUUID) { result in
            switch result {
            case .success(let data):
                // we got the data, let's convert to a string
                SSID = String(bytes: data.subdata(in: 0..<data.count), encoding: .utf8)
            case .failure(let error):
                finishWithResult(.failure(error))
            }
            reads.leave()
        }

        reads.enter()
        readData(from: CBUUID(string: "B5F90003-AA8D-11E3-9046-0002A5D5C51B"), serviceUUID: serviceUUID) { result in
            switch result {
            case .success(let data):
                // we got the data, let's convert to a string
                password = String(bytes: data.subdata(in: 0..<data.count), encoding: .utf8)
            case .failure(let error):
                finishWithResult(.failure(error))
            }
            reads.leave()
        }

        reads.notify(queue: .main) {
            guard let SSID = SSID, let password = password else {
                finishWithResult(.failure(CameraError.invalidResponse))
                return
            }

            finishWithResult(.success(WiFiSettings(SSID: SSID, password: password)))
        }
    }
    
    /// Reads camera's status

    func requestCameraStatus(_ completion: ((Result<CameraStatus, Error>) -> Void)?) {
        let serviceUUID = CBUUID(string: "FEA6")
        let commandUUID = CBUUID(string: "B5F90076-AA8D-11E3-9046-0002A5D5C51B")
        let commandResponseUUID = CBUUID(string: "B5F90077-AA8D-11E3-9046-0002A5D5C51B")
        let data = Data([0x07, 0x13, 0x08, 0x11, 0x2B, 0x2C, 0x37, 0x60])

        let finishWithResult: (Result<CameraStatus, Error>) -> Void = { result in
            // make sure to dispatch the result on the main thread
            DispatchQueue.main.async {
                completion?(result)
            }
        }

        registerObserver(serviceUUID: serviceUUID, characteristicUUID: commandResponseUUID) { data in

            var messageHexString = ""
            for i in 0 ..< data.count {
                messageHexString += String(format: "%02X", data[i])
            }
            NSLog("Status Raw: \(messageHexString)")
            
            // The response to the command is expected to be 12 bytes
            if data.count < 18 {
                finishWithResult(.failure(CameraError.invalidResponse))
                return
            }
            var cmode: UInt8 = 0x00
            if (data[12] == 0x60){
                cmode = data[17]
            } else if(data[12] == 0x2C){
                switch (data[11]){
                case 0x00:
                    //Video
                    cmode = 0xE8
                    if(data[14] == 0x01){
                        cmode = 0xEA
                    }
                case 0x01:
                    //Photo
                    cmode = 0xE8
                case 0x02:
                    //Timelapse
                    cmode = 0xEA
                default:
                    cmode = 0x00
                }
            }
            finishWithResult(.success(CameraStatus(busy: (data[5] == 0x01), mode: cmode, wifiEnabled: (data[8] == 0x01), previewAvailable: (data[11] == 0x01), openGoPro: (data[12] == 0x60))))
            return
        } completion: { [weak self] error in
            // Check that we successfully enable the notification for the response before writing to the characteristic
            if error != nil { finishWithResult(.failure(error!)); return }
            self?.write(data: data, serviceUUID: serviceUUID, characteristicUUID: commandUUID) { error in
                if error != nil { finishWithResult(.failure(error!)) }
            }
        }
    }
    
}
