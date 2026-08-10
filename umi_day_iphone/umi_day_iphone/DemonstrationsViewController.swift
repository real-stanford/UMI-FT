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

class DemonstrationsViewController: UIViewController, UITableViewDataSource, UITableViewDelegate, UIDocumentPickerDelegate {
    
    @IBOutlet weak var titleLabel: UILabel!
    @IBOutlet weak var qrCalibrationLabel: UILabel!
    @IBOutlet weak var gripperCalibrationLabel: UILabel!
    @IBOutlet weak var tableView: UITableView!
    var fnames: [String] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // load all the data
        do {
            fnames = try DemonstrationData.listDemonstrations()
        } catch {
           print("failed to load demo data")
        }
        
        // Step 3: Set data source and delegate
        tableView.dataSource = self
        tableView.delegate = self
        
        // Register UITableViewCell class or a custom cell
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")

        // Step 4: Add the table view to the view hierarchy
        self.view.addSubview(tableView)
        
        updateLabels()
    }

    // Step 5: Implement required UITableViewDataSource methods

    // Returns the number of rows in a section
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return fnames.count // Your data count here
    }

    // Returns the cell for a specific row at an index path
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        cell.textLabel?.text = "\(fnames[indexPath.row])" // Configure the cell
        return cell
    }

    // Optional: Handle table view row selection
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        let secondVC = storyboard.instantiateViewController(identifier: "ViewDemonstrationFileController") as ViewDemonstrationFileViewController
        
        secondVC.modalPresentationStyle = .fullScreen
        secondVC.modalTransitionStyle = .crossDissolve

        present(secondVC, animated: true, completion: nil)
        secondVC.initialize(demonstrationName: fnames[indexPath.row])
        
        tableView.deselectRow(at: indexPath, animated: true)
    }
    
    func tableView(_ tableView: UITableView, commit editingStyle: UITableViewCell.EditingStyle, forRowAt indexPath: IndexPath) {
        if editingStyle == .delete {
            // Remove the item from the data array
            do {
                try DemonstrationData.discard(recordingName: fnames[indexPath.row])
                fnames = try DemonstrationData.listDemonstrations()
                updateLabels()
                // Remove the row from the table view with an animation
                tableView.deleteRows(at: [indexPath], with: .automatic)
            } catch {
                print("failed to delete demonstration")
            }
        }
    }
    
    func updateLabels() {
        // update QR and gripper calibration run names
        let defaults = UserDefaults.standard
        var qrCalibrationRunName = (defaults.object(forKey: "qrCalibrationRunName") as? String)!
        var gripperCalibrationRunName = (defaults.object(forKey: "gripperCalibrationRunName") as? String)!
        var foundQRCalibration = fnames.contains(qrCalibrationRunName)
        var foundGripperCalibration = fnames.contains(gripperCalibrationRunName)
        
        if !foundQRCalibration {
            defaults.set("", forKey: "qrCalibrationRunName")
            qrCalibrationRunName = ""
        }
        if !foundGripperCalibration {
            defaults.set("", forKey: "gripperCalibrationRunName")
            gripperCalibrationRunName = ""
        }
        
        qrCalibrationLabel.text = "QR Calibration: \(qrCalibrationRunName)"
        gripperCalibrationLabel.text = "Gripper Calibration: \(gripperCalibrationRunName)"
        
        // update demonstration count title
        var numDemonstrations = 0
        for fname in fnames {
            if fname.contains("_demonstration_") {
                numDemonstrations += 1
            }
        }
        titleLabel.text = "\(numDemonstrations) Demonstrations"
    }

    @IBAction func exportButtonPressed(_ sender: Any) {
        // Create a document picker to save files
//        let documentPicker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
        let documentPicker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder], asCopy: false)


        documentPicker.delegate = self
        documentPicker.modalPresentationStyle = .formSheet
        present(documentPicker, animated: true, completion: nil)
    }
    
    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let selectedFolderURL = urls.first else { return }
        
        do {
            var demonstrationNames = try DemonstrationData.listDemonstrations()
            
            if !demonstrationNames.isEmpty {
                for demonstationName in demonstrationNames {
                    // Use NSFileCoordinator to handle file operations
                    let coordinator = NSFileCoordinator()

                    coordinator.coordinate(writingItemAt: selectedFolderURL, options: [.forReplacing], error: nil) { newURL in
                        // Access the security scoped resource
                        if selectedFolderURL.startAccessingSecurityScopedResource() {
                            defer { selectedFolderURL.stopAccessingSecurityScopedResource() }
                            
                            // Create the output folder that contains the date, ex: "2025-02-10"
                            
                            let exportRootFolder = "UMI_iPhone"
                            let dateDir = String(demonstationName[..<demonstationName.firstIndex(of: "T")!])// everything up to the date
                            
                            // Define the new file URL within the selected folder
                            let outputURL = newURL.appending(path: exportRootFolder, directoryHint: .isDirectory).appending(path: dateDir, directoryHint: .isDirectory).appending(component: demonstationName, directoryHint: .isDirectory)

                            coordinator.coordinate(writingItemAt: selectedFolderURL, options: [.forReplacing], error: nil) { newURL in
                                // Access the security scoped resource
                                if selectedFolderURL.startAccessingSecurityScopedResource() {
                                    defer { selectedFolderURL.stopAccessingSecurityScopedResource() }
                                    
                                    // Write the JSON data to the new file URL
                                    do {
                                        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true, attributes: nil)
                                        print("Created export folder")
                                    } catch {
                                        print("Error creating export folder: \(error)")
                                    }
                                } else {
                                    print("Failed to access the security scoped resource")
                                }
                            }

                            // write the demonstartion to the selected URL
                            do {
                                try DemonstrationData.saveExternally(recordingName: demonstationName, directoryURL: outputURL)
                                print("Demonstration saved to: \(outputURL)")
                            } catch {
                                print("Error writing JSON file: \(error)")
                            }
                        } else {
                            print("Failed to access the security scoped resource")
                        }
                    }
                }
                
                // show confirmation message that demonstrations were sucessfully saved
                let alertController = UIAlertController(title: "Successfully saved demonstrations", message: "", preferredStyle: .alert)
                let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
                    alertController.dismiss(animated: true, completion: nil)
                }
                alertController.addAction(okAction)
                
                self.present(alertController, animated: true, completion: nil)
            } else {
                print("Skipped export since there is no demonstration data")
                
                // show confirmation message that no demonstrations were saved
                let alertController = UIAlertController(title: "No demonstrations to save", message: "", preferredStyle: .alert)
                let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
                    alertController.dismiss(animated: true, completion: nil)
                }
                alertController.addAction(okAction)
                
                self.present(alertController, animated: true, completion: nil)
            }
        } catch {
            print("Failed to write demonstrations")
            
            // show confirmation message that no demonstrations were saved
            let alertController = UIAlertController(title: "Failed to write demonstrations", message: "", preferredStyle: .alert)
            let okAction = UIAlertAction(title: "Dismiss", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
            }
            alertController.addAction(okAction)
            
            self.present(alertController, animated: true, completion: nil)
        }
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        // Handle cancellation
        print("Document picker was cancelled")
    }
    
    @IBAction func backButtonPressed(_ sender: Any) {
        self.dismiss(animated: true)
    }
    
    @IBAction func deleteAllButtonPressed(_ sender: Any) {
        let alert = UIAlertController(title: "Delete all demonstrations", message: "Are you sure you want to delete all demonstrations (NOT RECOVERABLE)?", preferredStyle: .alert)

        // You can add actions using the following code
        alert.addAction(UIAlertAction(title: NSLocalizedString("Cancel", comment: "This closes alert"), style: .default, handler: { _ in
        NSLog("The \"OK\" alert occured.")
        }))
        alert.addAction(UIAlertAction(title: NSLocalizedString("Delete", comment: "This deletes all demonstrations"), style: .default, handler: { _ in
            for (index, fname) in self.fnames.enumerated() {
                do {
                    try DemonstrationData.discardDemonstrationsDir()
                } catch {
                    print("failed to delete demonstration")
                }
            }
            
            self.fnames = []
            self.updateLabels()
            self.tableView.reloadData()
            
        }))

        // This part of code inits alert view
        self.present(alert, animated: true, completion: nil)
    }
}
