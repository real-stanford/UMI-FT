//
//  RootViewController.swift
//  umi_day_iphone
//
//  Created by Austin Patel on 1/31/25.
//  Copyright © 2025 Apple. All rights reserved.
//

import UIKit

class RootViewController: UIViewController {
        
    override func viewDidLoad() {

    }
    
    override func viewDidAppear(_ animated: Bool) {
        let defaults = UserDefaults.standard
        var appMode = defaults.object(forKey: "appMode") as? String
        if appMode == nil {
            appMode = "demonstration"
            defaults.set(appMode, forKey: "appMode")
        }
        
        if appMode == "demonstration" {
            collectDemonstration()
        } else {
            assert(appMode == "deployment")
            startDeployment()
        }
    }
    
    func collectDemonstration() {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        let secondVC = storyboard.instantiateViewController(identifier: "DemonstrationViewController")

        secondVC.modalPresentationStyle = .fullScreen
        secondVC.modalTransitionStyle = .crossDissolve

        present(secondVC, animated: true, completion: nil)
    }
    
    func startDeployment() {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        let secondVC = storyboard.instantiateViewController(identifier: "DeploymentViewController")

        secondVC.modalPresentationStyle = .fullScreen
        secondVC.modalTransitionStyle = .crossDissolve

        present(secondVC, animated: true, completion: nil)
    }
}
