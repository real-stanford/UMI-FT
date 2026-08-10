//
//  NarrationCommands.swift
//  umi_day_iphone
//
//  Created by Austin Patel on 1/14/25.
//  Copyright © 2025 Apple. All rights reserved.
//

class NarrationCommands {
    
    public static func isStartWord(_ word: String) -> Bool {
        var result = false
        word.split(separator: " ").forEach {word in
            if word.lowercased() == "start" || word.lowercased() == "begin" {
                result = true
            }
        }
        return result
    }
    
    public static func isDoneWord(_ word: String) -> Bool {
        var result = false
        word.split(separator: " ").forEach {word in
            if word.lowercased() == "done" || word.lowercased() == "stop" {
                result = true
            }
        }
        return result
    }
    
    public static func isFinishedWord(_ word: String) -> Bool {
        var result = false
        word.split(separator: " ").forEach {word in
            if word.lowercased() == "finish" || word.lowercased() == "finished" {
                result = true
            }
        }
        return result
    }
    
}
