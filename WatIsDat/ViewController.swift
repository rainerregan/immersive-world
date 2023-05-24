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
import SceneKit.ModelIO

class ViewController: UIViewController {

    /// `sceneView` is an instance of ARSCNView which is a view that displays 3D augmented reality content. `debugText` is an instance of UITextView which is a view that displays text content. The `@IBOutlet` keyword indicates that the variable is an outlet that can be connected to a user interface element in Interface Builder.
    @IBOutlet var sceneView: ARSCNView!
    @IBOutlet weak var debugText: UITextView!
    
    /// `ViewControllerDelegate` is a protocol in Swift programming language that defines methods that can be implemented by a delegate of a view controller¹. A delegate is an object that acts on behalf of another object. It is used to handle events or modify the behavior of the view controller². The delegate methods defined in `ViewControllerDelegate` can be used to customize the behavior of a view controller. For example, you can use it to pass data between view controllers³.
    var viewControllerDelegate: ViewControllerDelegate!
    
    /// A  variable containing the latest CoreML prediction
    var latestPrediction : String = "lorem"
    
    // CoreML
    var visionRequests = [VNRequest]()
    let dispatchQueueML = DispatchQueue(label: "com.hw.dispatchqueueml") // A Serial Queue
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        /// Set the view's delegate
        viewControllerDelegate = ViewControllerDelegate()
        sceneView.delegate = viewControllerDelegate
        
        /// Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        /// Create a new scene
        let scene = SCNScene(named: "art.scnassets/ship.scn")!
        
        /// Set the scene to the view
        sceneView.scene = scene
        
        // MARK: - TAP GESTURE RECOGNIZER
        
        /// Set the tap gesture recognizer
        let tapGesture = UITapGestureRecognizer(
            target: self,
            action: #selector(self.handleTap(gestureRecognizer:))
        )
        view.addGestureRecognizer(tapGesture)
        
        // MARK: - Vision Model Config
        
        /// It declares a constant variable defaultMLConfig which is an instance of MLModelConfiguration. MLModelConfiguration is a class that provides configuration options for an ML model. The code then loads a Core ML model called HandDrawingModel_New using the VNCoreMLModel class. The VNCoreMLModel class is used to load a Core ML model into a Vision request. The loaded model is assigned to the selectedModel constant variable. If there is an error loading the model, the code will print an error message and terminate the program.
        let defaultMLConfig = MLModelConfiguration();
        guard let selectedModel = try? VNCoreMLModel(for: HandDrawingModel_New(configuration: defaultMLConfig).model) else {
            fatalError("Error on loading ML Model")
        }
        
        /// It declares a constant variable classificationRequest which is an instance of VNCoreMLRequest. VNCoreMLRequest is a class that performs image analysis using a Core ML model. The selectedModel is passed as a parameter to the VNCoreMLRequest initializer. The completionHandler parameter is set to classificationCompleteHandler, which is a function that will be called when the request completes. The code then sets the imageCropAndScaleOption property of the request to VNImageCropAndScaleOption.centerCrop. This option crops and scales the image to fit the input size required by the model. Finally, the request is added to an array called visionRequests.
        let classificationRequest = VNCoreMLRequest(model: selectedModel, completionHandler: classificationCompleteHandler)
        classificationRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop // Crop the image
        visionRequests = [classificationRequest]
        
        /// Loop for updating CoreML
        loopCoreMLUpdate()
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        /// Create a session configuration
        let configuration = ARWorldTrackingConfiguration()

        /// Run the view's session
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        /// Pause the view's session
        sceneView.session.pause()
    }
    
    // MARK: - Interaction Configuration
    
    /// The handleTap function is responsible for handling tap gestures on the AR scene view. It performs a raycast query to determine the intersection point between the tap location and any estimated planes in the scene. If an intersection is found, a 3D model is loaded based on a prediction and placed at the intersection point in the scene. This function enables users to place 3D models by tapping on the AR scene.
    @objc func handleTap(gestureRecognizer : UITapGestureRecognizer) {
        
        /// Create a raycast query using the current frame
        if let raycastQuery: ARRaycastQuery = sceneView.raycastQuery(
            from: gestureRecognizer.location(in: self.sceneView),
            allowing: .estimatedPlane,
            alignment: .any
        ) {
            // Performing raycast from the clicked location
            let raycastResults: [ARRaycastResult] = sceneView.session.raycast(raycastQuery)
            
            // Based on the raycast result, get the closest intersecting point on the plane
            if let closestResult = raycastResults.first {
                /// Get the coordinate of the clicked location
                let transform : matrix_float4x4 = closestResult.worldTransform
                let worldCoord : SCNVector3 = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
                
                /// Load 3D Model into the scene as SCNNode and adding into the scene
                guard let node : SCNNode = loadNodeBasedOnPrediction(latestPrediction) else {return}
                sceneView.scene.rootNode.addChildNode(node)
                node.position = worldCoord
            }
        }

        
    }
    
    /// The loadNodeBasedOnPrediction function loads a 3D model node based on a provided text prediction. It expects the text to be a valid filename for a USDZ file. The function first constructs a URL path using the provided text, trims whitespace and newline characters, and appends the "usdz" file extension. If the URL path cannot be created, indicating a missing file, the function returns nil.
    /// If the URL path is valid, an MDLAsset is created using the URL path. Textures associated with the model are loaded, and the first object from the asset is extracted. A new SCNNode is created using the extracted object, and its scale is set to 0.001 in all dimensions.
    /// Finally, the function returns the created SCNNode representing the loaded 3D model, or nil if any step fails.
    func loadNodeBasedOnPrediction(_ text: String) -> SCNNode? {
        guard let urlPath = Bundle.main.url(forResource: text.trimmingCharacters(in: .whitespacesAndNewlines), withExtension: "usdz") else {
            return nil
        }
        let mdlAsset = MDLAsset(url: urlPath)
        mdlAsset.loadTextures()
        
        let asset = mdlAsset.object(at: 0) // extract first object
        let assetNode = SCNNode(mdlObject: asset)
        assetNode.scale = SCNVector3(0.001, 0.001, 0.001)
        assetNode.eulerAngles = SCNVector3Make(.pi/2, 0, 0);
        
        return assetNode
    }
    
    // MARK: - CoreML Vision Handling
    
    /// The loopCoreMLUpdate function is responsible for continuously updating CoreML predictions in a loop. It is designed to be executed on a separate dispatch queue (dispatchQueueML) to prevent blocking the main thread.
    /// Inside the function, the updateCoreML method is called to perform the CoreML update, which likely involves processing input data and obtaining predictions from a CoreML model.
    /// After the CoreML update is completed, the function calls itself recursively, effectively creating a loop. This recursive call ensures that the loopCoreMLUpdate function will continue running repeatedly, continuously updating CoreML predictions.
    /// By executing the function on a separate dispatch queue and recursively calling itself, the loopCoreMLUpdate function provides a way to continuously update and use CoreML predictions in real-time applications or processes.
    func loopCoreMLUpdate() {
        dispatchQueueML.async {
            // Update CoreML
            self.updateCoreML()
            
            // Loop this function
            self.loopCoreMLUpdate()
        }
    }
    
    /// This is a Swift code that uses CoreML and Vision frameworks. It captures the current frame from the camera and converts it into a `CIImage`. Then it prepares a CoreML Vision request using `VNImageRequestHandler` and runs the image request using `perform(_:)` method. The `visionRequests` is an array of `VNCoreMLRequest` objects that are used to process the image. The code is used for classifying images with Vision and Core ML¹.
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
    
    /// This code is written in Swift programming language. It is a function that takes two parameters: a request and an error. The function first checks if there is an error and prints it if there is one. Then it gets the classification from the request results. It gets the top 2 results and maps them to a string with the identifier and confidence of each classification observation. The string is then joined with a new line separator. The function then updates the debug text on the screen with the classification string. It also stores the latest prediction by getting the object name from the classification string. The object name is obtained by splitting the classification string by "-" and "," characters and getting the first element of the resulting array.
    /// The code uses VNClassificationObservation which is a type of observation that results from performing a VNCoreMLRequest image analysis with a Core ML model whose role is classification¹.
    func classificationCompleteHandler(request: VNRequest, error: Error?) {
        /// Catch Errors
        if error != nil {
            print("Error: " + (error?.localizedDescription)!)
            return
        }
        guard let observations = request.results else {
            print("No results")
            return
        }
        
        /// Get the clasification, take the top 2 result of the prediction, and create the string display value
        let classifications = observations[0...1] // Get top 2 results
            .compactMap({ $0 as? VNClassificationObservation })
            .map({ "\($0.identifier) \(String(format: "- %.2f", $0.confidence))" }) // Confidence
            .joined(separator: "\n")
        
        DispatchQueue.main.async {
            // Print the clasification
//            print(classifications)
//            print("---------")
            
            /// Display Debug Text on screen
            var debugText:String = ""
            debugText += classifications
            self.debugText.text = debugText
            
            /// Store the latest prediction
            var objectName:String = "default"
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
