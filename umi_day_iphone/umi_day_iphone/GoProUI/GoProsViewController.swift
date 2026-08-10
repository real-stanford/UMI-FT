//
//  DemonstrationsViewController.swift
//  umi_day_iphone
//
//  Created by Austin Patel on 9/13/24.
//  Copyright © 2024 Apple. All rights reserved.
//

import Foundation
import MobileCoreServices
import UIKit
import CoreBluetooth

protocol GoProControllerDelegate {
    func setGoPro(_ gopro: GoPro)
    func disconnectGoPro()
}

class GoProsViewController: UIViewController, UITableViewDataSource, UITableViewDelegate, UIDocumentPickerDelegate {
    
    @IBOutlet weak var tableView: UITableView!
    var scanner = CentralManager()
    private var gopro: GoPro?
    private var connectingToName: String?
    
    var delegate : GoProControllerDelegate?
    
    @IBOutlet weak var startCaptureButton: UIButton!
    @IBOutlet weak var stopCaptureButton: UIButton!
    @IBOutlet weak var sleepButton: UIButton!
    
    var goProUIElements: [UIView] = []
    
    private let notificationCenter = NotificationCenter.default
    
    private var nothingSelectedYet = true
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Step 3: Set data source and delegate
        tableView.dataSource = self
        tableView.delegate = self
        
        // Register UITableViewCell class or a custom cell
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")

        // Step 4: Add the table view to the view hierarchy
        self.view.addSubview(tableView)
        
        goProUIElements = [startCaptureButton, stopCaptureButton, sleepButton]
        hideGoProControls()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        NSLog("Scanning for GoPro cameras..")
        scanner.start(withServices: [CBUUID(string: "FEA6")])
        notificationCenter.addObserver(self, selector:#selector(goProFound), name: NSNotification.Name("GoProFound"), object: nil)
        notificationCenter.addObserver(self, selector:#selector(goProDisconnect), name: NSNotification.Name("GoProDisconnect"), object: nil)
    }
    
    @objc func goProFound() {
        print("go pro found")
        
        
        if nothingSelectedYet {
            let defaults = UserDefaults.standard
            var savedGoProName = defaults.object(forKey: "recordingGoPro") as? String
            
            if let savedGoProName = savedGoProName {
                for peripheral in scanner.peripherals {
                    if peripheral.name == savedGoProName {
                        print("Connecting to GoPro set in user defaults")
                        connectingToName = savedGoProName
                        connectTo(peripheral: peripheral)
                    }
                }
            }
        }
        
        updateDisplay()
    }
    
    @objc func goProDisconnect() {
        print("go pro disconnect")
        
        scanner.stop()
        scanner.start(withServices: [CBUUID(string: "FEA6")])
        connectingToName = nil
        
        updateDisplay()
        hideGoProControls()
    }
    
    @objc func updateDisplay(){
        if self.viewIfLoaded?.window != nil {
            tableView.reloadData()
            print("reload table data")
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        notificationCenter.removeObserver(self)
        scanner.stop()
    }

    // Step 5: Implement required UITableViewDataSource methods

    // Returns the number of rows in a section
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return scanner.peripherals.count
    }

    // Returns the cell for a specific row at an index path
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        print("Cell for row at \(indexPath.row) and connectingToName is \(connectingToName)")
        
        var peripheralsSorted = scanner.peripherals.sorted { $0.name < $1.name }
        
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        var goproName = peripheralsSorted[indexPath.row].name
        var label = goproName
        if let gopro = gopro {
            if gopro.name == label {
                label += " (Connected)"
            }
        } else {
            if connectingToName != nil && connectingToName! == goproName {
                label += " (Connecting)"
            }
        }
        
        cell.textLabel?.text = label // Configure the cell
        return cell
    }
    
    func connectTo(peripheral: Peripheral) {
        connectingToName = peripheral.name
        updateDisplay()
        
        peripheral.connect { error in
            if error != nil {
                NSLog("Error connecting to \(peripheral.name)")
//                        self.gopro?.disconnect()
            } else {
                NSLog("Connected to \(peripheral.name)!")
                self.gopro = GoPro(peripheral)
            }
            self.connectingToName = nil
            self.updateDisplay()
            self.showGoProControls()
        }
    }

    // Optional: Handle table view row selection
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        print("row selected. Num peripherals is \(scanner.peripherals.count)")
        if (scanner.peripherals.count > 0){
            nothingSelectedYet = false
            var peripheralsSorted = scanner.peripherals.sorted { $0.name < $1.name }
            let selected: Peripheral = peripheralsSorted[indexPath.row]
            if selected.name == gopro?.name {
                // if connected already, then disconnect
                selected.disconnect()
                gopro = nil
                let defaults = UserDefaults.standard
                defaults.removeObject(forKey: "recordingGoPro")
                delegate?.disconnectGoPro()
                print("disconnected from \(selected.name)")
//                self.updateDisplay()
            } else {
                // if not connected to any others then connect
                if gopro != nil {
                    alertError(title: "Another GoPro is already connected", message: "Please disconnect from the first GoPro before connecting to this one.")
                } else {
                    print("attempting to connect to \(selected.name)")
                    connectTo(peripheral: selected)
                }
            }
            
            
        }
        
        tableView.deselectRow(at: indexPath, animated: true)
    }
    
    func alertError(title: String, message: String) {
        DispatchQueue.main.async {
            // Present the error that occurred.
            let alertController = UIAlertController(title: title, message: message, preferredStyle: .alert)
            let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
            }
            alertController.addAction(okAction)
            self.present(alertController, animated: true, completion: nil)
        }
    }
    
    @IBAction func startCaptureButtonPress(_ sender: Any) {
        gopro?.startRecording()
    }
    
    @IBAction func stopCaptureButtonPress(_ sender: Any) {
        gopro?.stopRecording()
    }
    
    @IBAction func backButtonPressed(_ sender: Any) {
        if let gopro = gopro {
            if let delegate = delegate {
                print("setting gopro in delegate")
                delegate.setGoPro(gopro)
            }
        }
        self.dismiss(animated: true)
    }
    
    @IBAction func sleepButtonPressed(_ sender: Any) {
        gopro?.sleep()
        gopro = nil
        connectingToName = nil
    }
    
    func showGoProControls() {
        for element in goProUIElements {
            element.isHidden = false
        }
    }
    
    func hideGoProControls() {
        for element in goProUIElements {
            element.isHidden = true
        }
    }
}
