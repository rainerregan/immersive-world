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

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate, ARSKViewDelegate {

    /// UIKit Components
    @IBOutlet weak var resButton: UIButton!
    @IBOutlet var sceneView: ARSCNView!
    @IBOutlet weak var debugText: UITextView!
    
    /// The ML model to be used for recognition of arbitrary objects.
    private var _handDrawingModel: HandDrawingModel_v4!
    private var handDrawingModel: HandDrawingModel_v4! {
        get {
            if let model = _handDrawingModel { return model }
            _handDrawingModel = {
                do {
                    let configuration = MLModelConfiguration()
                    return try HandDrawingModel_v4(configuration: configuration)
                } catch {
                    fatalError("Couldn't create HandDrawingModel due to: \(error)")
                }
            }()
            return _handDrawingModel
        }
    }
    
    /// A  variable containing the latest CoreML prediction
    private var identifierString = ""
    private var confidence: VNConfidence = 0.0
    private let dispatchQueueML = DispatchQueue(label: "com.exacode.dispatchqueueml") // A Serial Queue
    private var currentBuffer : CVImageBuffer?
    
    @IBAction func resButtonAction() {
        print("Reset")
        guard let sceneView = sceneView else {return}
        sceneView.scene.rootNode.enumerateChildNodes { (node, stop) in
        node.removeFromParentNode() }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        sceneView.delegate = self
        sceneView.session.delegate = self
        sceneView.showsStatistics = true
        sceneView.preferredFramesPerSecond = 10
        
        /// Create a new scene
        let scene = SCNScene(named: "art.scnassets/ship.scn")!
        sceneView.scene = scene
        
        // MARK: - TAP GESTURE RECOGNIZER
        /// Set the tap gesture recognizer
        let tapGesture = UITapGestureRecognizer(
            target: self,
            action: #selector(self.handleTap(gestureRecognizer:))
        )
        view.addGestureRecognizer(tapGesture)
        
        // MARK: - Start the scanning session
        self.restartSession()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        /// Create a session configuration
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]

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
            alignment: .horizontal
        ) {
            // Performing raycast from the clicked location
            let raycastResults: [ARRaycastResult] = sceneView.session.raycast(raycastQuery)
            
            print(raycastResults.debugDescription)
            
            // Based on the raycast result, get the closest intersecting point on the plane
            if let closestResult = raycastResults.first {
                /// Get the coordinate of the clicked location
                let transform : matrix_float4x4 = closestResult.worldTransform
                let worldCoord : SCNVector3 = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
                
                /// Load 3D Model into the scene as SCNNode and adding into the scene
                guard let node : SCNNode = loadNodeBasedOnPrediction(identifierString) else {return}
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
//        assetNode.eulerAngles = SCNVector3Make(.pi/2, 0, 0);
        
        return assetNode
    }
    
    // MARK: - CoreML Vision Handling
    
    private lazy var classificationRequest: VNCoreMLRequest = {
        do {
            // Instantiate the model from its generated Swift class.
            let model = try VNCoreMLModel(for: handDrawingModel.model)
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.classificationCompleteHandler(request: request, error: error)
            })
            
            // Crop input images to square area at center, matching the way the ML model was trained.
            request.imageCropAndScaleOption = .centerCrop
            
            // Use CPU for Vision processing to ensure that there are adequate GPU resources for rendering.
            request.usesCPUOnly = true
            
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard currentBuffer == nil, case .normal = frame.camera.trackingState else {
            return
        }
        
        // Retain the image buffer for Vision processing.
        self.currentBuffer = frame.capturedImage
        classifyCurrentImage();
    }
    
    func classifyCurrentImage() {
        
        let imageRequestHandler = VNImageRequestHandler(
            cvPixelBuffer: currentBuffer!,
            orientation: .up
        )
        
        // Run Image Request
        dispatchQueueML.async {
            do {
                // Release the pixel buffer when done, allowing the next buffer to be processed.
                defer { self.currentBuffer = nil }
                try imageRequestHandler.perform([self.classificationRequest])
            } catch {
                print(error)
            }
        }
    }
    
    func classificationCompleteHandler(request: VNRequest, error: Error?) {
        guard let results = request.results else {
            print("Unable to classify image.\n\(error!.localizedDescription)")
            return
        }
        
        // The `results` will always be `VNClassificationObservation`s, as specified by the Core ML model in this project.
        let classifications = results as! [VNClassificationObservation]
        
        // Show a label for the highest-confidence result (but only above a minimum confidence threshold).
        if let bestResult = classifications.first(where: { result in result.confidence > 0.5 }),
            let label = bestResult.identifier.split(separator: ",").first {
            identifierString = String(label)
            confidence = bestResult.confidence
        } else {
            identifierString = ""
            confidence = 0
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.displayClassifiedResult()
        }
    }
    
    func displayClassifiedResult() {
        // Print the clasification
        print("Clasification: \(self.identifierString)", "Confidence: \(self.confidence)")
        print("---------")
        
        self.debugText.text = "I'm \(self.confidence * 100)% sure this is a/an \(self.identifierString)"
    }
    
    private func restartSession() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }
    
    // MARK: - ARSCNViewDelegate

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
