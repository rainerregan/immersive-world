//
//  ViewController.swift
//  WatIsDat
//
//  Created by Rainer Regan on 24/05/23.
//

import UIKit
import SceneKit
import ARKit
import Vision

class ViewController: UIViewController {

    @IBOutlet var sceneView: ARSCNView!
    @IBOutlet weak var debugText: UITextView!
    
    var viewControllerDelegate: ViewControllerDelegate!
    
    let bubbleDepth : Float = 0.01 // the 'depth' of 3D text
    var latestPrediction : String = "THIS IS SPARTA" // a variable containing the latest CoreML prediction
    
    // CoreML
    var visionRequests = [VNRequest]()
    let dispatchQueueML = DispatchQueue(label: "com.hw.dispatchqueueml") // A Serial Queue
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set the view's delegate
        viewControllerDelegate = ViewControllerDelegate()
        sceneView.delegate = viewControllerDelegate
        
        // Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        // Create a new scene
        let scene = SCNScene(named: "art.scnassets/ship.scn")!
        
        // Set the scene to the view
        sceneView.scene = scene
        
        // MARK: - TAP GESTURE RECOGNIZER
        let tapGesture = UITapGestureRecognizer(
            target: self,
            action: #selector(self.handleTap(gestureRecognizer:))
        )
        view.addGestureRecognizer(tapGesture)
        
        // MARK: - Vision Model Config
        let defaultMLConfig = MLModelConfiguration();
        guard let selectedModel = try? VNCoreMLModel(for: Resnet50(configuration: defaultMLConfig).model) else {
            fatalError("Error on loading ML Model")
        }
        
        // Setup VisionCoreML request
        let classificationRequest = VNCoreMLRequest(model: selectedModel, completionHandler: classificationCompleteHandler)
        classificationRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop // Crop the image
        visionRequests = [classificationRequest]
        
        // Loop for updating CoreML
        loopCoreMLUpdate()
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()

        // Run the view's session
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    // MARK: - Interaction Configuration
    @objc func handleTap(gestureRecognizer : UITapGestureRecognizer) {
        
        // Create a raycast query using the current frame
        if let raycastQuery: ARRaycastQuery = sceneView.raycastQuery(
            from: gestureRecognizer.location(in: self.sceneView),
            allowing: .estimatedPlane,
            alignment: .any
        ) {
            // Perform the raycast and get the results
            let raycastResults: [ARRaycastResult] = sceneView.session.raycast(raycastQuery)
            
            // Get the closest result from the hit
            if let closestResult = raycastResults.first {
                // Get Coordinates of HitTest
                let transform : matrix_float4x4 = closestResult.worldTransform
                let worldCoord : SCNVector3 = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
                
                // Create 3D Text
                let node : SCNNode = createNewBubbleParentNode(latestPrediction)
                sceneView.scene.rootNode.addChildNode(node)
                node.position = worldCoord
            }
        }

        
    }
    
    func createNewBubbleParentNode(_ text: String) -> SCNNode {
        // TEXT BILLBOARD CONSTRAINT
        let billboardConstraint = SCNBillboardConstraint()
        billboardConstraint.freeAxes = SCNBillboardAxis.Y
        
        // BUBBLE-TEXT
        let bubble = SCNText(string: text, extrusionDepth: CGFloat(bubbleDepth))
        let font = UIFont(name: "Arial", size: 0.15)
        bubble.font = font
        bubble.alignmentMode = "center"
        bubble.firstMaterial?.diffuse.contents = UIColor.orange
        bubble.firstMaterial?.specular.contents = UIColor.white
        bubble.firstMaterial?.isDoubleSided = true
        bubble.chamferRadius = CGFloat(bubbleDepth)
        
        // BUBBLE NODE
        let (minBound, maxBound) = bubble.boundingBox
        let bubbleNode = SCNNode(geometry: bubble)
        // Centre Node - to Centre-Bottom point
        bubbleNode.pivot = SCNMatrix4MakeTranslation( (maxBound.x - minBound.x)/2, minBound.y, bubbleDepth/2)
        // Reduce default text size
        bubbleNode.scale = SCNVector3Make(0.2, 0.2, 0.2)
        
        // CENTRE POINT NODE
        let sphere = SCNSphere(radius: 0.005)
        sphere.firstMaterial?.diffuse.contents = UIColor.cyan
        let sphereNode = SCNNode(geometry: sphere)
        
        // BUBBLE PARENT NODE
        let bubbleNodeParent = SCNNode()
        bubbleNodeParent.addChildNode(bubbleNode)
        bubbleNodeParent.addChildNode(sphereNode)
        bubbleNodeParent.constraints = [billboardConstraint]
        
        return bubbleNodeParent
    }

    // MARK: - CoreML Vision Handling
    
    func loopCoreMLUpdate() {
        dispatchQueueML.async {
            // Update CoreML
            self.updateCoreML()
            
            // Loop this function
            self.loopCoreMLUpdate()
        }
    }
    
    func updateCoreML() {
        // Get camera image
        let pixbuff : CVPixelBuffer? = (sceneView.session.currentFrame?.capturedImage)
        if pixbuff == nil {return}
        let ciImage = CIImage(cvImageBuffer: pixbuff!)
        
        // Prepare CoreML Vision Request
        let imageRequestHandler = VNImageRequestHandler(
            ciImage: ciImage,
            options: [:]
        )
        
        // Run Image Request
        do {
            try imageRequestHandler.perform(self.visionRequests)
        } catch {
            print(error)
        }
    }
    
    /// Classification Complete Handler
    /// Untuk menghandle classification model dari CoreML
    func classificationCompleteHandler(request: VNRequest, error: Error?) {
        // Catch Errors
        if error != nil {
            print("Error: " + (error?.localizedDescription)!)
            return
        }
        guard let observations = request.results else {
            print("No results")
            return
        }
        
        // Get the clasification
        let classifications = observations[0...1] // Get top 2 results
//            .filter({
//                $0.confidence >= 0.5
//            })
            .flatMap({ $0 as? VNClassificationObservation })
            .map({ "\($0.identifier) \(String(format: "- %.2f", $0.confidence))" }) // Confidence
            .joined(separator: "\n")
        
        DispatchQueue.main.async {
            // Print the clasification
            print(classifications)
            print("---------")
            
            // Display Debug Text on screen
            var debugText:String = ""
            debugText += classifications
            self.debugText.text = debugText
            
            // Store the latest prediction
            var objectName:String = "NOPE"
            objectName = classifications.components(separatedBy: "-")[0]
            objectName = objectName.components(separatedBy: ",")[0]
            self.latestPrediction = objectName
            
        }
    }
}

// MARK: - ARSCNViewDelegate

class ViewControllerDelegate: NSObject, ARSCNViewDelegate {
    
/*
    // Override to create and configure nodes for anchors added to the view's session.
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        let node = SCNNode()
     
        return node
    }
*/
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user
        
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        // Inform the user that the session has been interrupted, for example, by presenting an overlay
        
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        // Reset tracking and/or remove existing anchors if consistent tracking is required
        
    }
}
