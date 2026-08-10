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

/// A simple wrapper around CBCentralManager to handle CoreBluetooth Central related tasks
final class CentralManager: NSObject {
    private let notificationCenter = NotificationCenter.default

    var peripherals: [Peripheral] = []

    private var manager: CBCentralManager!
    private var isReady: Bool { get { return manager.state == .poweredOn } }
    @Atomic private var onCentralStateChange: ((CBManagerState) -> Void)?

    override init() {
        super.init()
        let queue = DispatchQueue(label: "com.gopro.ble.central.queue", qos: .default)
        manager = CBCentralManager(delegate: self, queue: queue)
    }

    /// Starts a new BLE scan
    /// - Parameter withServices: the service UUIDs to scan for
    func start(withServices: [CBUUID]) {
        if isReady {
            peripherals.removeAll()
            manager.scanForPeripherals(withServices: withServices, options: nil)
        } else {
            onCentralStateChange = { [weak self] state in
                if state == .poweredOn {
                    DispatchQueue.main.async {
                        self?.start(withServices: withServices)
                    }
                }
            }
        }
    }

    /// Stops the current scan
    func stop() {
        manager.stopScan()
    }

    /// Connects the peripheral
    /// - Parameter peripheral: The peripheral to be connected
    func connectPeripheral(_ peripheral: CBPeripheral) {
        manager.connect(peripheral, options: nil)
    }

    /// Disconnects the peripheral
    /// - Parameter peripheral: The peripheral to disconnect
    func disconnectPeripheral(_ peripheral: CBPeripheral) {
        manager.cancelPeripheralConnection(peripheral)
    }
}

extension CentralManager : CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        onCentralStateChange?(central.state)
    }

    func centralManager(_ central: CBCentralManager, didDiscover cbPeripheral: CBPeripheral,
                        advertisementData: [String : Any], rssi RSSI: NSNumber) {
        guard let localName: String = advertisementData["kCBAdvDataLocalName"] as? String else { return }

        let peripheral = Peripheral(peripheral: cbPeripheral, localName: localName, manager: self)
        if peripherals.filter({$0.identifier == peripheral.identifier}).first == nil {
            DispatchQueue.main.async { [weak self] in
                self?.peripherals.append(peripheral)
                self!.notificationCenter.post(name: Notification.Name("GoProFound"), object: nil)
            }
        }
    }

    func centralManager(_ central: CBCentralManager, didConnect cbPeripheral: CBPeripheral) {
        guard let peripheral = peripherals.filter({$0.identifier == cbPeripheral.identifier.uuidString}).first else { return }
        peripheral.onConnect?(nil)
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect cbPeripheral: CBPeripheral, error: Error?) {
        guard let peripheral = peripherals.filter({$0.identifier == cbPeripheral.identifier.uuidString}).first,
              let error = error else { return }

        peripheral.onConnect?(error)

        DispatchQueue.main.async { [weak self] in
            let index = self?.peripherals.firstIndex(of: peripheral)
            self?.peripherals.remove(at: index!)
            self!.notificationCenter.post(name: Notification.Name("GoProDisconnect"), object: nil)
        }

    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral cbPeripheral: CBPeripheral, error: Error?) {
        guard let peripheral = peripherals.filter({$0.identifier == cbPeripheral.identifier.uuidString}).first else { return }

        peripheral.onDisconnect?(error)

        DispatchQueue.main.async { [weak self] in
            let index = self?.peripherals.firstIndex(of: peripheral)
            self?.peripherals.remove(at: index!)
            self!.notificationCenter.post(name: Notification.Name("GoProDisconnect"), object: nil)
        }
    }
}
