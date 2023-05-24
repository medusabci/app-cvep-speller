// MEDUSA-PLATFORM 
// v2023.0 GAIA
// www.medusabci.com

// c-VEP Speller (Unity app)
//      > Author: Víctor Martínez-Cagigal

// Versions:
//      - v1.0 (19/05/2022):    Circular-shifting c-VEP speller working
//      - v1.1 (04/07/2022):    Fixed small bug in which the app displayed and additional trial in training
//      - v2.0 (19/05/2023):    Checkerboard and manual stimulus parameters added

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;


public class Manager : MonoBehaviour
{
    // Public parameters
    public string IP = "127.0.0.1";
    public int port = 50000;

    public float fpsResolution = 60;    // Screen refresh rate (frequency bin, resolution) in Hz
    public int tau = 4;                 // Lag between consecutive commands (in bits)
    public int nRows = 4;
    public int nCols = 4;
    public int nSeqs = 1;
    public int currentMatrixIdx = 0;
    public int trainCycles = 10;
    public int trainTrials = 5;
    public int testCycles = 5;

    public float tPrevText = 1.0f;
    public float tPrevIddle = 0.5f;
    private float tFinishText = 1.0f;

    private MessageInterpreter.ParameterDecoder parameters = null;

    // Others
    private float minSeparatorSize = 40;

    // MEDUSA RUN STATES
    const int RUN_STATE_READY = 0;           // READY
    const int RUN_STATE_RUNNING = 1;         // RUNNING
    const int RUN_STATE_PAUSED = 2;          // PAUSED
    const int RUN_STATE_STOP = 3;            // TRANSITORY STATE WHILE USER PRESS THE STOP BUTTON AND MEDUSA IS READY TO START A NEW RUN AGAIN
    const int RUN_STATE_FINISHED = 4;        // THE RUN IS STILL ACTIVE, BUT FINISHED

    // Inner states
    const int STATE_WAITING_CONNECTION = -2;    // states of "state"
    const int STATE_WAITING_PARAMS = -1;

    const int STATE_WAITING_MODEL = 15;
    const int STATE_TRANSITION_TRAIN_TEST = 16;

    const int STATE_WAITING_SELECTION = 18;
    const int STATE_SELECTION_RECEIVED = 19;
    const int STATE_SELECTION_IDDLE = 20;

    const int STATE_RUNNING_PREVTEXT = 10;      // states of "innerstate" of innerRunningCycle()
    const int STATE_RUNNING_IDDLE = 11;
    const int STATE_RUNNING_TARGET = 12;
    const int STATE_RUNNING_IDDLE2 = 13;
    const int STATE_RUNNING_FLICKERING = 14;

    const int STATE_FINISHING_IDDLE = 25;       // states of "finishingstate" of finishingCycle()
    const int STATE_FINISHING_TEXT = 26;

    const int STATE_CLOSING_TEXT = 27;          // states of "closingstate" of closingApplication()
    const int STATE_CLOSING_FINAL = 28;
    const int STATE_EVERYTHING_CLOSED = 29;

    const int STATE_TRANSITION_TEXT = 30;       // states of "transitionstate" of transitionFastMode()
    const int STATE_TRANSITION_IDDLE = 31;

    const int STATE_RESULT_SHOW = 50;           // states of "resultstate" of showingResult()
    const int STATE_RESULT_IDDLE = 51;
    const int STATE_RESULT_END = 52;

    // State controllers and coroutines
    static int state = STATE_WAITING_CONNECTION;
    static int innerstate = STATE_RUNNING_PREVTEXT;
    static int finishingstate = STATE_FINISHING_IDDLE;
    static int closingstate = STATE_CLOSING_TEXT;
    static int resultstate = STATE_RESULT_SHOW;
    static bool mustStartTrial = false;
    static bool mustFinishRun = false;
    static bool mustClose = false;
    static bool mustShowResult = false;

    // Colors
    public Color32 defaultBoxColor = new Color32(255, 255, 255, 255);
    public Color32 highlightBoxColor = new Color32(75, 75, 75, 255);
    public Color32 targetBoxColor = new Color32(255, 25, 91, 255);
    public Color32 defaultTextColor = new Color32(183, 183, 183, 255);
    public Color32 highlightTextColor = new Color32(255, 255, 255, 255);
    public Color32 highlightResultBoxColor = new Color32(3, 252, 90, 255);
    public Color32 backgroundColor = new Color32(183, 183, 183, 255);
    public Color32 goodFPSColor = new Color32(94, 229, 125, 255);
    public Color32 badFPSColor = new Color32(180, 50, 40, 255);
    public Color32 resultBoxColor = new Color32(140, 140, 140, 255);
    public Color32 resultLabelColor = new Color32(183, 183, 183, 255);
    public Color32 resultTextColor = new Color32(244, 246, 87, 255);

    // Stimuli
    private Color32[] colorsBox;
    private Color32[] colorsText;
    private bool isCheckerboard;
    public Sprite[] checkerboards;
    private float currStimSize, currStimSeparation;
    private float stimSizeStep = 10.0f;
    private float stimSeparationStep = 10.0f;
    private bool isResponsive = true;

    // FPS counter
    private float updateCount = 0;
    private float fixedUpdateCount = 0;
    private float updateUpdateCountPerSecond;
    private float updateFixedUpdateCountPerSecond;

    // Other attributes
    private List<List<int[]>> matrixTrainItemSequence, matrixTestItemSequence;
    private int matrixTestCurrentTimeShift, matrixTrainCurrentTimeShift;
    private Dictionary<int, Color32> cellColorsByValue, textColorsByValue;
    private string[,] matrixTrainLabels, matrixTestLabels;
    private MessageInterpreter.ParameterDecoder.BothMatrices matrices;
    private int matrixTestSequenceLength, matrixTrainSequenceLength;

    private Vector2 lastScreenSize;
    static int currentTestTarget = 0;
    static int currentTrainTarget = 0;
    static int currentTrainSequence = 0;
    static bool mustHighlightTarget = true;
    private MessageInterpreter messageInterpreter = new MessageInterpreter();
    private int[] sequence;
    private int[,] sequences, target_coords;
    private int[,] matrixTrainLags, matrixTrainLagsInit, matrixTrainSequenceIdx;
    private Camera mainCamera;
    private Canvas mainCanvas;
    private GameObject[,] matrixTest, matrixTrain;
    private GameObject fpsMonitorText, informationBox, informationText, mainTestCell, mainTrainCell, resultBox, resultText, debugText, photodiodeCell;
    private float width, height;
    private int[,] testTarget, trainTarget;
    private string mode;
    private bool targetsAvailable;
    private bool photodiodeEnabled;
    private int cycleTestCounter = 0;
    private int cycleTrainCounter = 0;
    private int[,] targetsRowCol;
    private int[] referenceCoordsTest = new int[3];
    private int[] referenceCoordsTrain = new int[3];
    private int[] lastResultCoords = new int[3];
    private string lastResult = "";

    // TCP client
    private MedusaTCPClient tcpClient;


    /* ----------------------------------- UNITY LIFE-CYCLE FUNCTIONS ------------------------------------ */

    void Awake()
    {
        lastScreenSize = new Vector2(Screen.width, Screen.height);

        if (!Application.isEditor)
        {
            // Take the IP and port from the arguments
            // Usage: c-VEP Speller.exe 127.0.0.1 50000
            string[] arguments = Environment.GetCommandLineArgs();
            IP = arguments[1];
            port = Int32.Parse(arguments[2]);
        }
            
    }

    // Start is called before the first frame update
    void Start()
    {
        // Start the TCP/IP server
        tcpClient = new MedusaTCPClient(this, IPAddress.Parse(IP), port);
        tcpClient.Start();

        // FPS monitoring
        fpsMonitorText = GameObject.Find("FPSmonitor");
        StartCoroutine(monitorFPS());

        // Information text and box
        informationBox = GameObject.Find("InformationBox");
        informationText = GameObject.Find("InformationText");

        // Result text and box
        resultBox = GameObject.Find("ResultBox");
        resultText = GameObject.Find("ResultText");

        // Hide main test and training cells 
        mainTestCell = GameObject.Find("Test_Cell_Main");
        mainTestCell.SetActive(false);
        mainTrainCell = GameObject.Find("Train_Cell_Main");
        mainTrainCell.SetActive(false);

        // Hide debug text
        debugText = GameObject.Find("DebugText");
        debugText.SetActive(false);

        // Find the photodiode cell object
        photodiodeCell = GameObject.Find("Photodiode_Cell");

        // WAIT until parameters are received!
        state = STATE_WAITING_CONNECTION;
    }

    // Show the FPS monitoring: current FPS and refresh rate
    void OnGUI()
    {
        fpsMonitorText.GetComponent<Text>().text = updateUpdateCountPerSecond.ToString() + " fps (@" + updateFixedUpdateCountPerSecond.ToString() + ")";
        if (updateFixedUpdateCountPerSecond < fpsResolution)
        {
            fpsMonitorText.GetComponent<Text>().color = badFPSColor;
        }
        else
        {
            fpsMonitorText.GetComponent<Text>().color = goodFPSColor;
        }
    }

    // This function quits the current application by stopping the TCP client and closing the window
    public void quitApplication()
    {
        if (tcpClient.socketConnection != null)
        {
            bool tcpClosed = tcpClient.Stop();
            if (tcpClosed)
            {
                Debug.Log("> MedusaTCPClient closed successfully!");
            }
        }
        Application.Quit();
    }

    public void quitApplicationFromException()
    {
        mustClose = true;
    }


    /* ------------------------------------- UPDATE FUNCTIONS -------------------------------------- */

    // Update call (FPS may vary)
    void Update()
    {
        updateCount += 1;

        /* MINIMUM RESOLUTION */
        if (Screen.width < 450 || Screen.height < 450)
        {
            Screen.SetResolution(450, 450, false);
        }

        /* KEYPRESSES AND MOUSE EVENTS */
        if ((state != RUN_STATE_RUNNING) & (!isResponsive))
        {
            float scrollInput = Input.GetAxis("Mouse ScrollWheel");
            if (scrollInput != 0)
            {
                Debug.Log("Detected scrollwheel: " + scrollInput);
                // Change stimulus dimensions: [Scroll + D]
                if(Input.GetKey(KeyCode.D))
                {
                    if (scrollInput > 0) currStimSize += stimSizeStep;
                    if (scrollInput < 0) currStimSize -= stimSizeStep;
                }

                // Change stimulus separation: [Scoll + S]
                if (Input.GetKey(KeyCode.S))
                {
                    if (scrollInput > 0) currStimSeparation += stimSeparationStep;
                    if (scrollInput < 0) currStimSeparation -= stimSeparationStep;
                }
                onScreenSizeChange((float)Screen.width, (float)Screen.height, parameters.isResponsive, currStimSize, currStimSeparation);
                debugText.GetComponent<Text>().text = $"Dimensions: {currStimSize} px\nSeparation: {currStimSeparation} px";
                debugText.SetActive(true);
            } 
        }
        

        /* BEHAVIOR FOR DIFFERENT STATES */
        // If the TCP client just connected, request the parameters
        if (state == STATE_WAITING_CONNECTION && tcpClient.isConnected())
        {
            state = STATE_WAITING_PARAMS;
            // If the connection have been just established, send the waiting flag
            ServerMessage sm = new ServerMessage("waiting");
            tcpClient.SendMessage(sm.ToJson());
        }

        // If we are waiting the parameters
        if (state == STATE_WAITING_PARAMS)
        {
            // If parameters have been already received, execute this in the main thread
            if (parameters != null)
            {
                onParametersReady();
            }
        }

        // If we have received a new selection, show it in the main thread
        if (state == STATE_SELECTION_RECEIVED)
        {
            if (resultstate == STATE_RESULT_SHOW)
            {
                // Show the result
                if (!String.IsNullOrEmpty(lastResult))
                {
                    concatenateNewResult(lastResult);
                    matrixTest[lastResultCoords[1], lastResultCoords[2]].GetComponent<Image>().color = highlightResultBoxColor;
                    lastResult = "";
                }
            }

            if (resultstate == STATE_RESULT_IDDLE)
            {
                // Default color
                matrixTest[lastResultCoords[1], lastResultCoords[2]].GetComponent<Image>().color = defaultBoxColor;
            }

            if (resultstate == STATE_RESULT_END)
            {
                // Start another trial?
                if (targetsAvailable)
                {
                    if (currentTestTarget >= testTarget.Length)
                    {
                        // If all the targets have been done, notify the server to finish the app
                        mustFinishRun = true;
                        state = RUN_STATE_FINISHED;
                    }
                }
                if (state != RUN_STATE_FINISHED)
                {
                    // Starting another trial
                    state = RUN_STATE_RUNNING;
                    innerstate = STATE_RUNNING_IDDLE;
                    mustStartTrial = true;
                }

                // Reset result
                lastResultCoords = new int[3];
                lastResult = "";
                resultstate = STATE_RESULT_SHOW;
            }

        }

        // If the run is finished
        if (state == RUN_STATE_FINISHED)
        {
            if (finishingstate == STATE_FINISHING_IDDLE)
            {
                setInformationText("");
            }
            if (finishingstate == STATE_FINISHING_TEXT)
            {
                setInformationText("Run finished");
            }
        }

        // If the Unity app is stopping (closing)
        if (state == RUN_STATE_STOP)
        {
            if (closingstate == STATE_CLOSING_TEXT)
            {
                setInformationText("Closing...");
            }
            if (closingstate == STATE_CLOSING_FINAL)
            {
                quitApplication();
                closingstate = STATE_EVERYTHING_CLOSED;
            }
        }

        /* EXECUTING CO-ROUTINES*/

        // If a new trial should be started, run the co-routine
        if (mustStartTrial)
        {
            debugText.SetActive(false);
            mustStartTrial = false;
            StartCoroutine(innerRunningCycle());
        }

        // If we must show the received result
        if (mustShowResult)
        {
            mustShowResult = false;
            StartCoroutine(showingResult());
        }

        // If the run must finish
        if (mustFinishRun)
        {
            // Show the finished text and notify
            mustFinishRun = false;
            StartCoroutine(finishingCycle());
        }

        // If the TCPServer must close
        if (mustClose)
        {
            if (tcpClient.socketConnection != null) 
            { 
                // Send the confirmation that the Unity's client is going to close
                ServerMessage sm = new ServerMessage("close");
                tcpClient.SendMessage(sm.ToJson());
            }

            // Close the application
            mustClose = false;         // Avoid sending it twice
            StartCoroutine(closingApplication());
        }
    }

    // Fixed update at fpsResolution (setted before in onParametersReady())
    void FixedUpdate()
    {
        fixedUpdateCount += 1;

        // Flicker the photodiode
        if (photodiodeEnabled)
        {
            if (photodiodeCell.GetComponent<Image>().color == highlightBoxColor)
            {
                photodiodeCell.GetComponent<Image>().color = defaultBoxColor;
            }
            else if (photodiodeCell.GetComponent<Image>().color == defaultBoxColor)
            {
                photodiodeCell.GetComponent<Image>().color = highlightBoxColor;
            }
        }

        // Train or Online?
        if (state == RUN_STATE_RUNNING && String.Equals(mode, "Train", StringComparison.OrdinalIgnoreCase))
        {
            loopTrain();
        }
        else if (state == RUN_STATE_RUNNING && String.Equals(mode, "Online", StringComparison.OrdinalIgnoreCase))
        {
            loopTest();
        }
    }

    // This function resets the train matrix by unflashing everything
    void resetTrainMatrix()
    {
        matrixTrainCurrentTimeShift = 0;
        for (int r = 0; r < matrixTrain.GetLength(0); r++)
        {
            for (int c = 0; c < matrixTrain.GetLength(1); c++)
            {
                matrixTrain[r, c].GetComponent<Image>().sprite = null;
                matrixTrain[r, c].GetComponent<Image>().color = defaultBoxColor;
                matrixTrain[r, c].transform.Find("CellText").gameObject.SetActive(true);
                matrixTrain[r, c].transform.Find("CellText").GetComponent<Text>().color = defaultTextColor;
                matrixTrain[r, c].transform.Find("CellPoint").gameObject.SetActive(false);
            }
        }
    }

    // This function resets the test matrix by unflashing everything
    void resetTestMatrix()
    {
        matrixTestCurrentTimeShift = 0;
        for (int r = 0; r < matrixTest.GetLength(0); r++)
        {
            for (int c = 0; c < matrixTest.GetLength(1); c++)
            {
                matrixTest[r, c].GetComponent<Image>().sprite = null;
                matrixTest[r, c].GetComponent<Image>().color = defaultBoxColor;
                matrixTest[r, c].transform.Find("CellText").gameObject.SetActive(true);
                matrixTest[r, c].transform.Find("CellText").GetComponent<Text>().color = defaultTextColor;
                matrixTest[r, c].transform.Find("CellPoint").gameObject.SetActive(false);
            }
        }
    } 

    // This function updates the position of the cells of the training and testing matrices
    //      Note: Automatic Unity's scaling messes everything up. If some items are positioned programatically, it is better
    //      to resize them using this function (e.g., cells, text size, etc.).
    void onScreenSizeChange(float width, float height, bool isAuto, float stimSize, float separation)
    {

        // Optimal proportions for texts
        float optimalWidth = 1365.0f;
        float optimalHeight = 768.0f;
        float optimalCellSize = 132.0f;

        // Resize the texts
        RectTransform rtInfoText = GameObject.Find("InformationText").GetComponent<RectTransform>();
        RectTransform rtResultText = GameObject.Find("ResultText").GetComponent<RectTransform>();
        RectTransform rtFPSMonitor = GameObject.Find("FPSmonitor").GetComponent<RectTransform>();
        RectTransform rtResultLabel = GameObject.Find("ResultLabel").GetComponent<RectTransform>();
        float sw_ = width / optimalWidth;
        float sh_ = height / optimalHeight;
        rtInfoText.localScale = new Vector3(sw_, sw_, 1f);
        rtResultText.localScale = new Vector3(sh_, sh_, 1f);
        rtFPSMonitor.localScale = new Vector3(sh_, sh_, 1f);
        rtResultLabel.localScale = new Vector3(sh_, sh_, 1f);

        // Substract the height of the result box
        Image resultBox = GameObject.Find("ResultBox").GetComponent<Image>();
        height -= resultBox.rectTransform.rect.height;

        // Automatic resizing
        float colSeparator = separation;
        float rowSeparator = separation;
        float cellSize = stimSize;
        if (isAuto)
        {
            // Compute the cell size
            float h_ = (height - (nRows + 1) * minSeparatorSize) / nRows;
            float w_ = (width - (nCols + 1) * minSeparatorSize) / nCols;
            cellSize = Mathf.Min(w_, h_);

            // Compute the separators
            colSeparator = (float)minSeparatorSize;
            rowSeparator = (float)minSeparatorSize;
            if (height < width)
            {
                colSeparator = (width - cellSize * nCols) / (nCols + 1);
            }
            else
            {
                rowSeparator = (height - cellSize * nRows) / (nRows + 1);
            }
        } 


        // TEST MATRIX

        // Compute the coordinate of the first cell 
        int testRows = matrixTest.GetLength(0);
        int testCols = matrixTest.GetLength(1);
        float x0 = (width - (cellSize * testCols + colSeparator * (testCols - 1))) / 2;
        float y0 = (height + (cellSize * testRows + rowSeparator * (testRows - 1))) / 2;
        Vector2 origin = new Vector2(x0, y0);

        // Move all cells
        for (int r = 0; r < matrixTest.GetLength(0); r++)
        {
            float y_ = origin.y - r * (cellSize + rowSeparator);
            for (int c = 0; c < matrixTest.GetLength(1); c++)
            {
                float x_ = origin.x + c * (cellSize + colSeparator);
                
                // Move and scale each command box
                RectTransform rt = matrixTest[r, c].GetComponent<RectTransform>();
                rt.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, cellSize);
                rt.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, cellSize);
                matrixTest[r, c].transform.position = new Vector2(x_, y_);

                // Adapt the text of each command label
                float st_ = cellSize / optimalCellSize;
                matrixTest[r, c].transform.Find("CellText").GetComponent<Text>().GetComponent<RectTransform>().localScale = new Vector3(st_, st_, 1f);
            }
        }

        // TRAIN MATRIX
        
        // Compute the coordinate of the first cell 
        int trainRows = matrixTrain.GetLength(0);
        int trainCols = matrixTrain.GetLength(1);
        x0 = (width - (cellSize * trainCols + colSeparator * (trainCols - 1))) / 2;
        y0 = (height + (cellSize * trainRows + rowSeparator * (trainRows - 1))) / 2;
        origin = new Vector2(x0, y0);

        // Move all cells (training matrix)
        for (int r = 0; r < trainRows; r++)
        {
            float y_ = origin.y - r * (cellSize + rowSeparator);
            for (int c = 0; c < trainCols; c++)
            {
                float x_ = origin.x + c * (cellSize + colSeparator);
                
                // Move and scale each command box
                RectTransform rt = matrixTrain[r, c].GetComponent<RectTransform>();
                rt.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, cellSize);
                rt.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, cellSize);
                matrixTrain[r, c].transform.position = new Vector2(x_, y_);

                // Adapt the text of each command label
                float st_ = cellSize / optimalCellSize;
                matrixTrain[r, c].transform.Find("CellText").GetComponent<Text>().GetComponent<RectTransform>().localScale = new Vector3(st_, st_, 1f);
            }
        }

        // Generate the checkerboards if required
        if (isCheckerboard)
        {
            checkerboards = generateCheckerboards((int)cellSize, parameters.stim_spatial_cycles, colorsBox[0], colorsBox[1]);
        }
    }
    
    // This function sets the information text. IMPORTANT: only the main thread is allowed to run this function.
    void setInformationText(string infoMsg)
    {
        if (string.IsNullOrEmpty(infoMsg))
        {
            informationBox.SetActive(false);
            informationText.SetActive(false);
        }
        else
        {
            informationBox.SetActive(true);
            informationText.SetActive(true);
            informationText.GetComponent<Text>().text = infoMsg;
        }
    }

    // This function controls the visibility of the training matrix
    void setTrainMatrixVisible(bool shouldBeVisible)
    {
        for (int r = 0; r < matrixTrain.GetLength(0); r++)
        {
            for (int c = 0; c < matrixTrain.GetLength(1); c++)
            {
                matrixTrain[r, c].SetActive(shouldBeVisible);
            }
        }
    }

    // This function controls the visibility of the test matrix
    void setTestMatrixVisible(bool shouldBeVisible)
    {
        for (int r = 0; r < matrixTest.GetLength(0); r++)
        {
            for (int c = 0; c < matrixTest.GetLength(1); c++)
            {
                matrixTest[r, c].SetActive(shouldBeVisible);
            }
        }
    }

    // This function concatenates a new result label to the result box
    void concatenateNewResult(string resultLabel)
    {
        resultText.GetComponent<Text>().text += resultLabel + " ";
    }

    /* ------------------------------------------- COMMUNICATION ------------------------------------------- */

    // This function is called by the MedusaTCPClient whenever a packet is received in order to interpret it
    public void interpretMessage(string message)
    {
        Debug.Log("Received from server: " + message);
        string eventType = messageInterpreter.decodeEventType(message);
        switch (eventType)
        {
            case "play":
                if (state != STATE_WAITING_PARAMS)
                {
                    state = RUN_STATE_RUNNING;
                    mustStartTrial = true;
                }
                break;
            case "pause":
                if (state != STATE_WAITING_PARAMS)
                    state = RUN_STATE_PAUSED;
                break;
            case "resume":
                if (state != STATE_WAITING_PARAMS)
                    state = RUN_STATE_RUNNING;
                break;
            case "stop":
                if (state != STATE_WAITING_PARAMS)
                {
                    state = RUN_STATE_STOP;
                    innerstate = STATE_RUNNING_PREVTEXT;
                    mustClose = true;
                }
                break;
            case "restart":
                if (state != STATE_WAITING_PARAMS)
                {
                    state = RUN_STATE_READY;
                    innerstate = STATE_RUNNING_PREVTEXT;
                }
                break;
            case "setParameters":
                // The main thread will detect that parameters are here using Update() and will call onParametersReady() itself
                parameters = messageInterpreter.decodeParameters(message);
                Debug.Log("Parameters received.");
                break;
            case "selection":
                // MEDUSA has selected a new command!
                int[] selection_coords = messageInterpreter.decodeSelection(message);
                onSelectedCommand(selection_coords);
                break;
            case "exception":
                string exception = messageInterpreter.decodeException(message);
                Debug.LogError("Exception from client, aborting: " + exception);
                state = RUN_STATE_STOP;
                innerstate = STATE_RUNNING_PREVTEXT;

                tcpClient.socketConnection.Close();
                tcpClient.socketConnection = null;
                mustClose = true;
                break;
            default:
                Debug.LogError("Unknown action!");
                break;
        }
    }

    // This function is called by the main thread when parameters are ready
    void onParametersReady()
    {
        // Extract the parameters
        mode = parameters.mode;
        fpsResolution = parameters.fpsResolution;
        tPrevText = parameters.tPrevText;
        tPrevIddle = parameters.tPrevIddle;
        tFinishText = parameters.tFinishText;
        photodiodeEnabled = parameters.photodiodeEnabled; // Using photodiode?
        trainCycles = parameters.trainCycles;
        trainTrials = parameters.trainTrials;
        testCycles = parameters.testCycles;
        matrices = parameters.matrices;
        currentMatrixIdx = 0;   // For now, only one matrix is supported

        // Set up the fixedDeltaTime to the desired frame rate for the clock
        Application.targetFrameRate = -1;
        Time.fixedDeltaTime = 1 / ((float)fpsResolution);

        // Hide or show the photodiode cell
        if (!photodiodeEnabled)
        {
            photodiodeCell.SetActive(false);
        }

        // Is checkerboard?
        if (String.Equals(parameters.stim_type, "checkerboard", StringComparison.OrdinalIgnoreCase))
        {
            isCheckerboard = true;
        }
            

        // Set up the default colors
        mainCamera = GameObject.FindGameObjectWithTag("MainCamera").GetComponent<Camera>();
        mainCamera.backgroundColor = hexToColor(parameters.color_background);
        mainCanvas = GameObject.Find("Canvas").GetComponent<Canvas>();

        Image resultBox = GameObject.Find("ResultBox").GetComponent<Image>();
        resultBox.color = hexToColor(parameters.color_result_info_box);
        GameObject.Find("ResultLabel").GetComponent<Text>().color = resultLabelColor;
        GameObject.Find("ResultText").GetComponent<Text>().color = resultTextColor;

        colorsBox = new Color32[] { hexToColor(parameters.color_box_0), hexToColor(parameters.color_box_1) };
        colorsText = new Color32[] { hexToColor(parameters.color_text_0), hexToColor(parameters.color_text_1) };

        // TEST MATRIX
        //      matrixTest                  -> Test matrix containing the GameObjects (i.e., each cell that contains the box and the text)
        //      matrixTestLabels            -> Label of each command
        //      matrixTestItemSequence      -> Matrix that contains the sequence of each cell
        //      matrixTestCurrentTimeShift  -> Counter that controls the position inside the encoding sequence at each monitor refresh

        // Find the matrix
        GameObject matrixTestObject = GameObject.FindGameObjectWithTag("MatrixTest");
        int testNRow = matrices.test[currentMatrixIdx].n_row;
        int testNCol = matrices.test[currentMatrixIdx].n_col;
        matrixTest = new GameObject[testNRow, testNCol];
        mainTestCell.SetActive(false);

        // Create the matrix by duplicating the first cell
        int idx = 0;
        matrixTestLabels = new string[testNRow, testNCol];
        matrixTestItemSequence = new List<List<int[]>> ();
        matrixTestCurrentTimeShift = 0;
        matrixTestSequenceLength = matrices.test[currentMatrixIdx].item_list[0].sequence.GetLength(0);   // All the items must have the same sequence length
        for (int r = 0; r < testNRow; r++)
        {
            matrixTestItemSequence.Add(new List<int[]>());
            for (int c = 0; c < testNCol; c++)
            {
                matrixTest[r, c] = Instantiate(mainTestCell, new Vector2(0, 0), new Quaternion(), matrixTestObject.transform);
                matrixTest[r, c].name = "Test_Cell_" + r.ToString() + "_" + c.ToString();
                matrixTest[r, c].transform.Find("CellText").GetComponent<Text>().text = matrices.test[currentMatrixIdx].item_list[idx].text;
                matrixTest[r, c].transform.Find("CellPoint").gameObject.SetActive(false);
                matrixTest[r, c].transform.Find("CellPoint").GetComponent<Image>().color = hexToColor(parameters.color_point);
                matrixTestItemSequence[r].Insert(c, matrices.test[currentMatrixIdx].item_list[idx].sequence);
                matrixTestLabels[r, c] = matrices.test[currentMatrixIdx].item_list[idx].text;
                idx++;
            }
        }

        // TRAINING MATRIX (only different sequences)

        // Find the matrix
        GameObject matrixTrainObject = GameObject.FindGameObjectWithTag("MatrixTrain");
        int trainNRow = matrices.train[currentMatrixIdx].n_row;
        int trainNCol = matrices.train[currentMatrixIdx].n_col;
        matrixTrain = new GameObject[trainNRow, trainNCol];
        mainTrainCell.SetActive(false);

        // Create the matrix by duplicating the first cell
        idx = 0;
        matrixTrainLabels = new string[trainNRow, trainNCol];
        matrixTrainItemSequence = new List<List<int[]>>();
        matrixTrainCurrentTimeShift = 0;
        matrixTrainSequenceLength = matrices.train[currentMatrixIdx].item_list[0].sequence.GetLength(0); ;   // All the items must have the same sequence length
        for (int r = 0; r < trainNRow; r++)
        {
            matrixTrainItemSequence.Add(new List<int[]>());
            for (int c = 0; c < trainNCol; c++)
            {
                matrixTrain[r, c] = Instantiate(mainTrainCell, new Vector2(0, 0), new Quaternion(), matrixTrainObject.transform);
                matrixTrain[r, c].name = "Train_Cell_" + r.ToString() + "_" + c.ToString();
                matrixTrain[r, c].transform.Find("CellText").GetComponent<Text>().text = matrices.train[currentMatrixIdx].item_list[idx].label;
                matrixTrain[r, c].transform.Find("CellPoint").gameObject.SetActive(false);
                matrixTrain[r, c].transform.Find("CellPoint").GetComponent<Image>().color = hexToColor(parameters.color_point);
                matrixTrainItemSequence[r].Insert(c, matrices.train[currentMatrixIdx].item_list[idx].sequence);
                matrixTrainLabels[r, c] = matrices.train[currentMatrixIdx].item_list[idx].label;
                idx++;
            }
        }

        // Detect what should be the initial matrix (training or test)
        if (String.Equals(mode, "Train", StringComparison.OrdinalIgnoreCase))
        {
            setTestMatrixVisible(false);
            setTrainMatrixVisible(true);
        }
        else
        {
            setTestMatrixVisible(true);
            setTrainMatrixVisible(false);
        }

        // Resize event to set up all positions
        isResponsive = parameters.isResponsive;
        currStimSeparation = (float)parameters.stim_separation;
        currStimSize = (float)parameters.stim_size;
        onScreenSizeChange((float)Screen.width, (float)Screen.height, isResponsive,
            currStimSize, currStimSeparation);

  
        // Change state
        state = RUN_STATE_READY;
        ServerMessage sm = new ServerMessage("ready");
        tcpClient.SendMessage(sm.ToJson());
        setInformationText("Waiting for start...");
    }

    // This function is called when a command is selected from MEDUSA
    void onSelectedCommand(int[] selectionCoords)
    {
        // Store the new result
        int idx = rowColToMatrixIndexTest(selectionCoords[0], selectionCoords[1], selectionCoords[2]);
        lastResult = matrices.test[selectionCoords[0]].item_list[idx].text;
        lastResultCoords = selectionCoords;
        state = STATE_SELECTION_RECEIVED;
        mustShowResult = true;
    }

    // This function returns the current timestamp in seconds from the Unix epoch (1/1/1970)
    double getCurrentTimeStamp()
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        long unixTimeMilliseconds = now.ToUnixTimeMilliseconds();
        double unixTimeSeconds = Convert.ToDouble(unixTimeMilliseconds) / 1000.0;
        return unixTimeSeconds;
    }

    // This function converts the coordinates of a row and column to the matrix index
    int rowColToMatrixIndexTest(int matrixIdx, int row, int col)
    {
        int idx = matrices.test[matrixIdx].n_col * row + col;
        return idx;
    }

    /* ----------------------------------------- TRAIN-TEST LOOPS ----------------------------------------- */

    // Loop for "Train" mode: calibration with all the m-sequences
    void loopTrain()
    {
        // First: show starting text
        if (innerstate == STATE_RUNNING_PREVTEXT)
        {
            setInformationText("Starting...");
        }
        // Second: standby 
        else if (innerstate == STATE_RUNNING_IDDLE)
        {
            setInformationText("");
        }
        // Target loop, first: Target shown
        else if (innerstate == STATE_RUNNING_TARGET)
        {
            matrixTrain[0, currentTrainSequence].GetComponent<Image>().color = targetBoxColor;
        }
        // Target loop, second: Standby
        else if (innerstate == STATE_RUNNING_IDDLE2 && mustHighlightTarget)
        {
            matrixTrain[0, currentTrainSequence].GetComponent<Image>().color = defaultBoxColor;
            mustHighlightTarget = false;
        }
        // Target loop, third: flickering
        else if (innerstate == STATE_RUNNING_FLICKERING)
        {
            // First: check if a new trial is starting
            if (matrixTrainCurrentTimeShift == 0)
            {
                // It is starting a new cycle, so the timestamp must be recorded and sent
                if (cycleTrainCounter < trainCycles)
                {
                    double currentTime = getCurrentTimeStamp();
                    ServerMessage sm = new ServerMessage("train");
                    sm.addValue("cycle", cycleTrainCounter);
                    sm.addValue("onset", currentTime);
                    sm.addValue("trial", currentTrainTarget); // TODO: SEVERAL TARGETS IN TRAINING
                    sm.addValue("matrix_idx", 0);
                    sm.addValue("unit_idx", 0);
                    sm.addValue("level_idx", 0);
                    sm.addValue("command_idx", 0);
                    sm.addValue("mode", "Train");
                    tcpClient.SendMessage(sm.ToJson());
                }
                // Important note: the previous IF statement prevents the system to send the onset when cycleTestCounter==testCycles, 
                // In such a way, the next stage lets the last cycle to be displayed completely. Otherwise, the last cycle onset 
                // would be sent to MEDUSA and immediately the flickering would stop, sending the "processPlease" command
                cycleTrainCounter++;
            }
            // Check how many cycles have been displayed
            if (cycleTrainCounter > trainCycles)
            {
                cycleTrainCounter = 0;
                resetTrainMatrix();

                // If all the targets have been done, notify the server
                if (currentTrainTarget >= trainTrials - 1)
                {
                    mustFinishRun = true;
                    state = RUN_STATE_FINISHED;
                }
                else
                {
                    // Start the next trial
                    currentTrainTarget++;
                    innerstate = STATE_RUNNING_IDDLE;
                    mustHighlightTarget = true;
                    mustStartTrial = true;          // Care: coroutines are not stopping if Stop(coroutine) is not called
                }          
            }
            else
            {   
                // Make the flashings
                for (int r = 0; r < matrixTrain.GetLength(0); r++)
                {
                    for (int c = 0; c < matrixTrain.GetLength(1); c++)
                    {
                        // Stimulus color/image
                        int value = matrixTrainItemSequence[r][c][matrixTrainCurrentTimeShift];
                        if (isCheckerboard)
                        {
                            matrixTrain[r, c].GetComponent<Image>().sprite = checkerboards[value];
                        } 
                        else
                        {
                            matrixTrain[r, c].GetComponent<Image>().color = colorsBox[value];
                        }
                        // Stimulus text
                        if (parameters.show_text)
                        {
                            matrixTrain[r, c].transform.Find("CellText").gameObject.SetActive(true);
                            matrixTrain[r, c].transform.Find("CellText").GetComponent<Text>().color = colorsText[value];
                        } else
                        {
                            matrixTrain[r, c].transform.Find("CellText").gameObject.SetActive(false);
                        }
                        // Stimulus midpoint
                        if (parameters.show_point) matrixTrain[r, c].transform.Find("CellPoint").gameObject.SetActive(true);
                    }
                }

                // Update the current index
                matrixTrainCurrentTimeShift++;
                if (matrixTrainCurrentTimeShift >= matrixTrainSequenceLength)
                {
                    matrixTrainCurrentTimeShift = 0;
                }
            }
        }
    }

    // Loop for "Online" mode: selecting items from the matrix
    void loopTest()
    {
        // First: show starting text
        if (innerstate == STATE_RUNNING_PREVTEXT)
        {
            setInformationText("Starting...");
        }
        // Second: standby 
        else if (innerstate == STATE_RUNNING_IDDLE)
        {
            setInformationText("");
        }
        // NOTE: the two first training steps are ignored as they are focused to highlight the target
        // Test loop, third: flickering
        else if (innerstate == STATE_RUNNING_FLICKERING)
        {
            // First: check if the reference is starting a new trial
            if (matrixTestCurrentTimeShift == 0)
            {
                // It is starting a new trial, so the timestamp must be recorded and sent
                if (cycleTestCounter < testCycles)
                {
                    double currentTime = getCurrentTimeStamp();
                    ServerMessage sm = new ServerMessage("test");
                    sm.addValue("cycle", cycleTestCounter);
                    sm.addValue("onset", currentTime);
                    sm.addValue("trial", currentTestTarget);
                    sm.addValue("mode", "Online");
                    sm.addValue("matrix_idx", currentMatrixIdx);
                    sm.addValue("unit_idx", 0);
                    sm.addValue("level_idx", 0);
                    tcpClient.SendMessage(sm.ToJson());
                }
                // Important note: the previous IF statement prevents the system to send the onset when cycleTestCounter==testCycles, 
                // In such a way, the next stage lets the last cycle to be displayed completely. Otherwise, the last cycle onset 
                // would be sent to MEDUSA and immediately the flickering would stop, sending the "processPlease" command
                cycleTestCounter++;
            }
            // Check how many cycles have been displayed
            if (cycleTestCounter > testCycles)
            {
                cycleTestCounter = 0;
                currentTestTarget++;
                resetTestMatrix();

                // Request MEDUSA to process the trial
                state = STATE_WAITING_SELECTION;
                ServerMessage sm = new ServerMessage("processPlease");
                tcpClient.SendMessage(sm.ToJson());
            }
            else
            {
                // Make the flashings
                for (int r = 0; r < matrixTest.GetLength(0); r++)
                {
                    for (int c = 0; c < matrixTest.GetLength(1); c++)
                    {
                        // Stimulus color/image
                        int value = matrixTestItemSequence[r][c][matrixTestCurrentTimeShift];
                        if (isCheckerboard)
                        {
                            matrixTest[r, c].GetComponent<Image>().sprite = checkerboards[value];
                        }
                        else
                        {
                            matrixTest[r, c].GetComponent<Image>().color = colorsBox[value];
                        }
                        // Stimulus text
                        if (parameters.show_text)
                        {
                            matrixTest[r, c].transform.Find("CellText").gameObject.SetActive(true);
                            matrixTest[r, c].transform.Find("CellText").GetComponent<Text>().color = colorsText[value];
                        } else
                        {
                            matrixTest[r, c].transform.Find("CellText").gameObject.SetActive(false);
                        }
                        // Stimulus midpoint
                        if (parameters.show_point) matrixTest[r, c].transform.Find("CellPoint").gameObject.SetActive(true);
                    }
                }

                // Update the current index
                matrixTestCurrentTimeShift++;
                if (matrixTestCurrentTimeShift >= matrixTestSequenceLength)
                {
                    matrixTestCurrentTimeShift = 0;
                }
            }

        }
    }

    /* ------------------------------------------- CO-ROUTINES ------------------------------------------- */

    // This thread controls the FPS rate
    IEnumerator monitorFPS()
    {
        while (true)
        {
            yield return new WaitForSeconds(1);
            updateUpdateCountPerSecond = updateCount;
            updateFixedUpdateCountPerSecond = fixedUpdateCount;

            updateCount = 0;
            fixedUpdateCount = 0;
        }
    }

    // This thread controls the timings of a running cycle. When the flashings start, this routine ends.
    IEnumerator innerRunningCycle()
    {
        if (innerstate <= STATE_RUNNING_PREVTEXT)
        {
            Debug.Log("Running: starting...");
            innerstate = STATE_RUNNING_PREVTEXT;
            yield return new WaitForSeconds((float)tPrevText);
        }

        if (innerstate <= STATE_RUNNING_IDDLE)
        {
            Debug.Log("Running: iddle.");
            innerstate = STATE_RUNNING_IDDLE;
            yield return new WaitForSeconds((float)tPrevIddle);
        }

        if (String.Equals(mode, "Train", StringComparison.OrdinalIgnoreCase))
        {
            if (innerstate <= STATE_RUNNING_TARGET)
            {
                Debug.Log("Running: target.");
                innerstate = STATE_RUNNING_TARGET;
                yield return new WaitForSeconds((float)tPrevText);
            }

            if (innerstate <= STATE_RUNNING_IDDLE2)
            {
                Debug.Log("Running: target iddle.");
                innerstate = STATE_RUNNING_IDDLE2;
                yield return new WaitForSeconds((float)tPrevIddle);
            }
        }

        if (innerstate <= STATE_RUNNING_FLICKERING)
        {
            Debug.Log("Running: flickering.");
            innerstate = STATE_RUNNING_FLICKERING;
        }
    }

    // This thread controls the timings of the finished run
    IEnumerator finishingCycle()
    {
        Debug.Log("Finishing...");
        finishingstate = STATE_FINISHING_IDDLE;
        yield return new WaitForSeconds((float)tPrevIddle);

        finishingstate = STATE_FINISHING_TEXT;
        yield return new WaitForSeconds((float)tFinishText);

        // Send the confirmation that the Unity's client has finished the execution
        // NOTE: we have waited the test to show to assure that enough samples after the last onset have been recorded in the MANAGER of MEDUSA
        ServerMessage sm = new ServerMessage("finish");
        tcpClient.SendMessage(sm.ToJson());
    }

    // This thread controls the timings for closing the application.
    IEnumerator closingApplication()
    {
        Debug.Log("Closing...");
        closingstate = STATE_CLOSING_TEXT;
        yield return new WaitForSeconds((float)tFinishText);

        closingstate = STATE_CLOSING_FINAL;
    }

    IEnumerator showingResult()
    {
        Debug.Log("Showing the result...");
        resultstate = STATE_RESULT_SHOW;
        yield return new WaitForSeconds((float)tPrevText);

        resultstate = STATE_RESULT_IDDLE;
        yield return new WaitForSeconds((float)tPrevIddle);

        resultstate = STATE_RESULT_END;
        Debug.Log("Showing result finished...");
    }

    /* ----------------------------------- CHECKERBOARD GENERATION --------------------------------------- */
    public static Sprite[] generateCheckerboards(int imageSize, int spatialCycles, Color32 color0, Color32 color1)
    {
        // Initialize the textures
        int blockSize = imageSize / (spatialCycles * 2);
        Texture2D texturePositive = new Texture2D(2 * spatialCycles * blockSize, 2 * spatialCycles * blockSize);
        Texture2D textureNegative = new Texture2D(2 * spatialCycles * blockSize, 2 * spatialCycles * blockSize);

        // Get the color arrays
        Color32[] colorArray0 = colorToArray(color0, blockSize * blockSize);
        Color32[] colorArray1 = colorToArray(color1, blockSize * blockSize);

        // Create the checkerboard pattern by painting group of pixels
        for (int i = 0; i < spatialCycles * 2; i++)
        {
            for (int j = 0; j < spatialCycles * 2; j++)
            {
                if (((i + j) % 2) == 1)
                {
                    texturePositive.SetPixels32(i * blockSize, j * blockSize, blockSize, blockSize, colorArray0);
                    textureNegative.SetPixels32(i * blockSize, j * blockSize, blockSize, blockSize, colorArray1);
                }
                else
                {
                    texturePositive.SetPixels32(i * blockSize, j * blockSize, blockSize, blockSize, colorArray1);
                    textureNegative.SetPixels32(i * blockSize, j * blockSize, blockSize, blockSize, colorArray0);
                }
            }
        }
        texturePositive.Apply();
        textureNegative.Apply();

        // Ignore spatial pixel interpolation
        texturePositive.wrapMode = TextureWrapMode.Clamp;
        texturePositive.filterMode = FilterMode.Point;
        textureNegative.wrapMode = TextureWrapMode.Clamp;
        textureNegative.filterMode = FilterMode.Point;

        // Create the sprites
        Sprite spritePositive = Sprite.Create(texturePositive, new Rect(0, 0, texturePositive.width, texturePositive.height), Vector2.one * 0.5f);
        Sprite spriteNegative = Sprite.Create(textureNegative, new Rect(0, 0, textureNegative.width, textureNegative.height), Vector2.one * 0.5f);
        return new Sprite[] { spritePositive, spriteNegative};
    }

    /* ------------------------------------------- COLOR UTILS ------------------------------------------- */

    public static string colorToHex(Color32 color)
    {
        string hex = color.r.ToString("X2") + color.g.ToString("X2") + color.b.ToString("X2");
        return hex;
    }

    public static Color hexToColor(string hex)
    {
        hex = hex.Replace("0x", "");//in case the string is formatted 0xFFFFFF
        hex = hex.Replace("#", "");//in case the string is formatted #FFFFFF
        byte a = 255;//assume fully visible unless specified in hex
        byte r = byte.Parse(hex.Substring(0, 2), System.Globalization.NumberStyles.HexNumber);
        byte g = byte.Parse(hex.Substring(2, 2), System.Globalization.NumberStyles.HexNumber);
        byte b = byte.Parse(hex.Substring(4, 2), System.Globalization.NumberStyles.HexNumber);
        //Only use alpha if the string has enough characters
        if (hex.Length == 8)
        {
            a = byte.Parse(hex.Substring(6, 2), System.Globalization.NumberStyles.HexNumber);
        }
        return new Color32(r, g, b, a);
    }

    public static Color32[] colorToArray(Color32 color, int arrayLength)
    {
        Color32[] colorArray = new Color32[arrayLength];
        for (int i = 0; i < arrayLength; i++)
        {
            colorArray[i] = new Color32(color.r, color.g, color.b, color.a);
        }
        return colorArray;
    }
}