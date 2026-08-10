//
//  ConnectGoPro.swift
//  umi_day_iphone
//
//  Created by Austin Patel on 9/25/24.
//  Copyright © 2024 Apple. All rights reserved.
//

import CoreBluetooth

class GoProManager {
    
    private let scanner = CentralManager()
    private let notificationCenter = NotificationCenter.default
    private var goProSearchName: String?
    private var foundHandler: ((_ gopro:GoPro) -> Void)?
    
    init() {
        notificationCenter.addObserver(self, selector:#selector(goProFound), name: NSNotification.Name("GoProFound"), object: nil)
    }
    
    func attemptConnection(goProName: String, foundHandler:@escaping (_ gopro:GoPro) -> Void) {
        goProSearchName = goProName
        self.foundHandler = foundHandler
        
        scanner.stop()
        scanner.start(withServices: [CBUUID(string: "FEA6")])
    }
    
    func stopScanning() {
        scanner.stop()
    }
    
    @objc func goProFound() {
        for peripheral in scanner.peripherals {
            if peripheral.name == goProSearchName {
                peripheral.connect { error in
                    if error != nil {
                        NSLog("Error connecting to \(peripheral.name)")
                    } else {
                        NSLog("Connected to \(peripheral.name)!")
                        self.foundHandler!(GoPro(peripheral))
                    }
                }
                
            }
        }
    }
}
