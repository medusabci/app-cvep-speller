// MEDUSA-PLATFORM 
// v2022.0 CHAOS
// www.medusabci.com

// c-VEP Speller (Unity app)
//      > Author: Víctor Martínez-Cagigal
//      > Date: 19/05/2022

// Versions:
//      - v1.0 (19/05/2022):    Circular-shifting c-VEP speller working
//      - v1.1 (04/07/2022):    Fixed small bug in which the app displayed and additional trial in training

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.IO;
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
    private List<List<int>> trainTargetCoords;
    public int testCycles = 5;
    public bool showPoint = true;
    public int pointSize = 8;

    public float tPrevText = 1.0f;
    public float tPrevIddle = 0.5f;
    public float tFinishText = 1.0f;

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
    public Color32 defaultTextColor = new Color32(183, 183, 183, 255);
    public Color32 highlightBoxColor = new Color32(75, 75, 75, 255);
    public Color32 targetBoxColor = new Color32(255, 25, 91, 255);
    public Color32 highlightResultBoxColor = new Color32(3, 252, 90, 255);
    public Color32 goodFPSColor = new Color32(94, 229, 125, 255);
    public Color32 badFPSColor = new Color32(180, 50, 40, 255);
    public Color32 resultBoxColor = new Color32(140, 140, 140, 255);
    public Color32 resultLabelColor = new Color32(183, 183, 183, 255);
    public Color32 resultTextColor = new Color32(244, 246, 87, 255);
    public Color32 pointColor = new Color32(128, 0, 0, 255);
    private Dictionary<int, Color32> textColorsByValue;
    private Dictionary<int, string> cellStimuliByValue;
    private Dictionary<int, Sprite> cellSpritesByValue;
    private Dictionary<int, int> cellOpacitiesByValue;

    // Scenarios
    private Image backgroundScenario;
    public string scenarioName;
    public string scenarioBlob;

    // FPS counter
    private float updateCount = 0;
    private float fixedUpdateCount = 0;
    private float updateUpdateCountPerSecond;
    private float updateFixedUpdateCountPerSecond;

    // Other attributes
    private List<List<int[]>> matrixItemSequence;
    private int matrixCurrentTimeShift;
    private string[,] matrixLabels;
    private List<MessageInterpreter.ParameterDecoder.Matrix> matrices;
    private int matrixSequenceLength;

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
    private GameObject[,] matrix;
    private GameObject fpsMonitorText, informationBox, informationText, debugText, mainCell, resultBox, resultText, photodiodeCell;
    private float cellSize;
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

    // Required for raster latencies (only works for Windows)
    [DllImport("user32.dll", EntryPoint = "FindWindow")]
    public static extern IntPtr FindWindow(System.String className, System.String windowName);
    [DllImport("user32.dll", EntryPoint = "GetWindowRect")]
    public static extern bool GetWindowRect(IntPtr hwnd, ref Rect rectangle);
    public int lastWindowLeft = 0;
    public int lastWindowTop = 0;

    /* ----------------------------------- GUI HELPERS  ------------------------------------ */

    private void changeItemColor(GameObject item, Color32 color)
    {
        item.GetComponent<Image>().color = color;
    }
    private void changeItemTexture(GameObject item, Sprite sprite)
    {
        item.GetComponent<Image>().sprite = sprite;
    }
    private void changeItemTextureOpacity(GameObject item,  int opacity)
    {
        Image image = item.GetComponent<Image>();
        float alpha = Mathf.Clamp01(opacity / 100f);
        image.color = new Color(1f, 1f, 1f, alpha);
    }
    private void changeItemTextColor(GameObject item, Color32 color)
    {
        item.transform.GetChild(0).GetComponent<Text>().color = color;
    }

    private void changeItemText(GameObject item, String text)
    {
        item.transform.GetChild(0).GetComponent<Text>().text = text;
    }
    private Text getItemText(GameObject item)
    {
        return item.transform.GetChild(0).GetComponent<Text>();
    }

    private void changeItemPointVisibility(GameObject item, bool mustBeVisible)
    {
        item.transform.GetChild(1).gameObject.SetActive(mustBeVisible);
    }

    private void changeItemPointColor(GameObject item, Color32 color)
    {
        item.transform.GetChild(1).GetComponent<Image>().color = color;
    }

    private void changeItemPointSize(GameObject item, int size)
    {
        item.transform.GetChild(1).GetComponent<RectTransform>().sizeDelta = new Vector2(size, size);
    }

    private Image getItemPoint(GameObject item)
    {
        return item.transform.GetChild(1).GetComponent<Image>();
    }

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

        // Hide main cells 
        mainCell = GameObject.Find("Cell_Main");
        mainCell.SetActive(false);

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

        /* RESIZE EVENT */
        Vector2 screenSize = new Vector2(Screen.width, Screen.height);
        if (this.lastScreenSize != screenSize)
        {
            this.lastScreenSize = screenSize;
            onScreenSizeChange((float)Screen.width, (float)Screen.height); //  Launch the event when the screen size change
        }
        else if (windowMoved())
        {
            notifyMovedItems();
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
                    changeItemColor(matrix[lastResultCoords[1], lastResultCoords[2]], highlightResultBoxColor);
                    changeItemTexture(matrix[lastResultCoords[1], lastResultCoords[2]], null);
                    changeItemTextColor(matrix[lastResultCoords[1], lastResultCoords[2]], defaultTextColor);
                    lastResult = "";
                }
            }

            if (resultstate == STATE_RESULT_IDDLE)
            {
                // Default color
                changeItemColor(matrix[lastResultCoords[1], lastResultCoords[2]], defaultBoxColor);
                changeItemTexture(matrix[lastResultCoords[1], lastResultCoords[2]], null);
                changeItemTextColor(matrix[lastResultCoords[1], lastResultCoords[2]], defaultTextColor);
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

    // This function resets the matrix by unflashing everything
    void resetMatrix()
    {
        matrixCurrentTimeShift = 0;
        for (int r = 0; r < matrix.GetLength(0); r++)
        {
            for (int c = 0; c < matrix.GetLength(1); c++)
            {
                changeItemColor(matrix[r, c], defaultBoxColor);
                changeItemTexture(matrix[r, c], null);
                changeItemTextColor(matrix[r, c], defaultTextColor);
            }
        }
    } 

    // This function updates the position of the cells of the training and testing matrices
    //      Note: Automatic Unity's scaling messes everything up. If some items are positioned programatically, it is better
    //      to resize them using this function (e.g., cells, text size, etc.).
    void onScreenSizeChange(float width, float height)
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

       
        // Compute the cell size
        float h_ = (height - (nRows + 1) * minSeparatorSize) / nRows;
        float w_ = (width - (nCols + 1) * minSeparatorSize) / nCols;
        cellSize = Mathf.Min(w_, h_);
        
        // Compute the separators
        float colSeparator = (float)minSeparatorSize;
        float rowSeparator = (float)minSeparatorSize;
        if (height < width)
        {
            colSeparator = (width - cellSize * nCols) / (nCols + 1);
        }
        else
        {
            rowSeparator = (height - cellSize * nRows) / (nRows + 1);
        }

        // MATRIX

        // Compute the coordinate of the first cell 
        int Rows = matrix.GetLength(0);
        int Cols = matrix.GetLength(1);
        float x0 = (width - (cellSize * Cols + colSeparator * (Cols - 1))) / 2;
        float y0 = (height + (cellSize * Rows + rowSeparator * (Rows - 1))) / 2;
        Vector2 origin = new Vector2(x0, y0);

        // Move all cells
        for (int r = 0; r < matrix.GetLength(0); r++)
        {
            float y_ = origin.y - r * (cellSize + rowSeparator);
            for (int c = 0; c < matrix.GetLength(1); c++)
            {
                float x_ = origin.x + c * (cellSize + colSeparator);
                
                // Move and scale each command box
                RectTransform rt = matrix[r, c].GetComponent<RectTransform>();
                rt.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, cellSize);
                rt.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, cellSize);
                matrix[r, c].transform.position = new Vector2(x_, y_);

                // Adapt the text of each command label
                float st_ = cellSize / optimalCellSize;
                getItemText(matrix[r, c]).GetComponent<RectTransform>().localScale = new Vector3(st_, st_, 1f);

                // Adapt the midpoint of each command box
                getItemPoint(matrix[r, c]).GetComponent<RectTransform>().localScale = new Vector3(st_, st_, 1f);
            }
        }

        // Notify MEDUSA about the position of each item for raster latencies correction
        notifyMovedItems();
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

    // This function controls the visibility of the matrix
    void setMatrixVisible(bool shouldBeVisible)
    {
        for (int r = 0; r < matrix.GetLength(0); r++)
        {
            for (int c = 0; c < matrix.GetLength(1); c++)
            {
                matrix[r, c].SetActive(shouldBeVisible);
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
        trainTargetCoords = parameters.trainTargetCoords;
        trainTrials =  trainTargetCoords.Count();
        testCycles = parameters.testCycles;
        showPoint = parameters.showPoint;
        pointSize = parameters.pointSize;
        matrices = parameters.matrices;
        cellStimuliByValue = parameters.stimulus_box_dict;
        cellOpacitiesByValue = parameters.opacity_box_dict;
        textColorsByValue = parameters.textColorsByValue;

        cellSpritesByValue = new Dictionary<int, Sprite>();

        foreach (KeyValuePair<int, string> blob in cellStimuliByValue)
        {
            int key = blob.Key;
            string base64Image = blob.Value;

            byte[] imageBytes = Convert.FromBase64String(base64Image);

            Texture2D texture = new Texture2D(2, 2);
            texture.LoadImage(imageBytes);

            float size = Mathf.Min((float)texture.width, texture.height);
            float x = (texture.width - size) * 0.5f;
            float y = (texture.height - size) * 0.5f;

            Sprite sprite = Sprite.Create(
                texture,
                new UnityEngine.Rect(x, y, size, size),
                new Vector2(0.5f, 0.5f)
            );

            cellSpritesByValue.Add(key, sprite);
        }

        currentMatrixIdx = 0;   // For now, only one matrix is supported

        // Set up the fixedDeltaTime to the desired frame rate for the clock
        Application.targetFrameRate = -1;
        Time.fixedDeltaTime = 1 / ((float)fpsResolution);

        // Hide or show the photodiode cell
        if (!photodiodeEnabled)
        {
            photodiodeCell.SetActive(false);
        }

        // Set up the default background
        mainCamera = GameObject.FindGameObjectWithTag("MainCamera").GetComponent<Camera>();
        mainCanvas = GameObject.Find("Canvas").GetComponent<Canvas>();
        backgroundScenario = GameObject.Find("Background Scenario").GetComponent<Image>();
        scenarioName = parameters.scenario_name;
        scenarioBlob = parameters.scenario_blob;
        if (String.Equals(scenarioName, "Solid Color", StringComparison.OrdinalIgnoreCase))
        {
            backgroundScenario.color = hexToColor(parameters.color_background);
        }
        else
        {
            byte[] imageBytes = Convert.FromBase64String(scenarioBlob);
            Texture2D texture = new Texture2D(2, 2);
            texture.LoadImage(imageBytes);
            float aspect = 1.6f;
            float tw = texture.width, th = texture.height;
            float texAspect = tw / th;

            float rw = tw, rh = th, rx = 0, ry = 0;
            if (texAspect > aspect)
            {
                rw = th * aspect;
                rx = (tw - rw) * 0.5f;
            }
            else
            {
                rh = tw / aspect;
                ry = (th - rh) * 0.5f;
            }

            backgroundScenario.sprite = Sprite.Create(
                texture,
                new UnityEngine.Rect(rx, ry, rw, rh),
                new Vector2(0.5f, 0.5f)
            );
        }

        // Set up the default colors
        resultBoxColor = hexToColor(parameters.color_result_info_box);
        resultLabelColor = hexToColor(parameters.color_result_info_label);
        resultTextColor = hexToColor(parameters.color_result_info_text);

        goodFPSColor = hexToColor(parameters.color_fps_good);
        badFPSColor = hexToColor(parameters.color_fps_bad);

        GameObject.Find("ResultBox").GetComponent<Image>().color = resultBoxColor;
        GameObject.Find("ResultLabel").GetComponent<Text>().color = resultLabelColor;
        GameObject.Find("ResultText").GetComponent<Text>().color = resultTextColor;

        targetBoxColor = hexToColor(parameters.color_target_box);

        highlightResultBoxColor = hexToColor(parameters.color_highlight_result_box);

        pointColor = hexToColor(parameters.color_point);

        // MATRIX
        //      matrix             -> Test matrix containing the GameObjects (i.e., each cell that contains the box and the text)
        //      matrixLabels            -> Label of each command
        //      matrixItemSequence      -> Matrix that contains the sequence of each cell
        //      matrixCurrentTimeShift  -> Counter that controls the position inside the encoding sequence at each monitor refresh

        // Find the matrix
        GameObject matrixObject = GameObject.FindGameObjectWithTag("Matrix");
        int NRow = matrices[currentMatrixIdx].n_row;
        int NCol = matrices[currentMatrixIdx].n_col;
        matrix = new GameObject[NRow, NCol];
        mainCell.SetActive(false);

        // Create the matrix by duplicating the first cell
        int idx = 0;
        matrixLabels = new string[NRow, NCol];
        matrixItemSequence = new List<List<int[]>> ();
        matrixCurrentTimeShift = 0;
        matrixSequenceLength = matrices[currentMatrixIdx].item_list[0].sequence.GetLength(0);   // All the items must have the same sequence length
        for (int r = 0; r < NRow; r++)
        {
            matrixItemSequence.Add(new List<int[]>());
            for (int c = 0; c <NCol; c++)
            {
                matrix[r, c] = Instantiate(mainCell, new Vector2(0, 0), new Quaternion(), matrixObject.transform);
                matrix[r, c].name = "Cell_" + r.ToString() + "_" + c.ToString();
                changeItemText(matrix[r, c], matrices[currentMatrixIdx].item_list[idx].text);
                changeItemPointColor(matrix[r, c], pointColor);
                changeItemPointSize(matrix[r, c], pointSize);
                changeItemPointVisibility(matrix[r, c], showPoint);
                matrixItemSequence[r].Insert(c, matrices[currentMatrixIdx].item_list[idx].sequence);
                matrixLabels[r, c] = matrices[currentMatrixIdx].item_list[idx].text;
                idx++;
            }
        }

        setMatrixVisible(true);

        // Resize event to set up all positions
        onScreenSizeChange((float)Screen.width, (float)Screen.height);

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
        int idx = rowColToMatrixIndex(selectionCoords[0], selectionCoords[1], selectionCoords[2]);
        lastResult = matrices[selectionCoords[0]].item_list[idx].text;
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
    int rowColToMatrixIndex(int matrixIdx, int row, int col)
    {
        int idx = matrices[matrixIdx].n_col * row + col;
        return idx;
    }

    // This function informs MEDUSA about the positions of each element across the screen for raster latency correction
    public void notifyMovedItems()
    {
        double currentTime = getCurrentTimeStamp();
        ServerMessage sm = new ServerMessage("resize");
        sm.addValue("resize_onset", currentTime);
        sm.addValue("screen_size", new int[] { Screen.currentResolution.width, Screen.currentResolution.height });

        ResizedMatrices resizedMatrices = new ResizedMatrices();
        // Training items
        int NRow = matrices[currentMatrixIdx].n_row;
        int NCol = matrices[currentMatrixIdx].n_col;
        int idx = 0;
        for (int r = 0; r < NRow; r++)
        {
            for (int c = 0; c < NCol; c++)
            {
                int[] coord = new int[] { currentMatrixIdx, matrices[currentMatrixIdx].item_list[idx].row, matrices[currentMatrixIdx].item_list[idx].col };
                RectTransform rt = matrix[r, c].GetComponent<RectTransform>();
                int curr_center_x = (int)(rt.position.x + rt.sizeDelta.x / 2);
                int curr_center_y = Screen.height - (int)(rt.position.y - rt.sizeDelta.y / 2);
                int[] new_pos = estimatePixelPosition(curr_center_x, curr_center_y);
                resizedMatrices.addItem(true, idx, coord, new_pos);
                idx++;
            }
        }
        sm.addValue("new_position", resizedMatrices);

        // Send the message
        tcpClient.SendMessage(sm.ToJson());
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
            int target_row = trainTargetCoords[currentTrainTarget][1];
            int target_col = trainTargetCoords[currentTrainTarget][2];

            changeItemColor(matrix[target_row, target_col], targetBoxColor);
            changeItemTexture(matrix[target_row, target_col], null);
            changeItemTextColor(matrix[target_row, target_col], defaultTextColor);
        }
        // Target loop, second: Standby
        else if (innerstate == STATE_RUNNING_IDDLE2 && mustHighlightTarget)
        {
            int target_row = trainTargetCoords[currentTrainTarget][1];
            int target_col = trainTargetCoords[currentTrainTarget][2];

            changeItemColor(matrix[target_row, target_col], defaultBoxColor);
            changeItemTexture(matrix[target_row, target_col], null);
            changeItemTextColor(matrix[target_row, target_col], defaultTextColor);
            mustHighlightTarget = false;
        }
        // Target loop, third: flickering
        else if (innerstate == STATE_RUNNING_FLICKERING)
        {
            // First: check if a new trial is starting
            if (matrixCurrentTimeShift == 0)
            {
                // It is starting a new cycle, so the timestamp must be recorded and sent
                if (cycleTrainCounter < trainCycles)
                {
                    int target_mtx = trainTargetCoords[currentTrainTarget][0];
                    int target_row = trainTargetCoords[currentTrainTarget][1];
                    int target_col = trainTargetCoords[currentTrainTarget][2];
                    int target_idx = matrices[target_mtx].n_col * target_row + target_col;

                    double currentTime = getCurrentTimeStamp();
                    ServerMessage sm = new ServerMessage("train");
                    sm.addValue("cycle", cycleTrainCounter);
                    sm.addValue("onset", currentTime);
                    sm.addValue("trial", currentTrainTarget); 
                    sm.addValue("matrix_idx",target_mtx);
                    sm.addValue("unit_idx", target_row);
                    sm.addValue("level_idx", target_col);
                    sm.addValue("command_idx", target_idx);
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
                resetMatrix();

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
                for (int r = 0; r < matrix.GetLength(0); r++)
                {
                    for (int c = 0; c < matrix.GetLength(1); c++)
                    {
                        int value = matrixItemSequence[r][c][matrixCurrentTimeShift];
                        if (cellSpritesByValue.ContainsKey(value))
                        {
                            changeItemTexture(matrix[r, c], cellSpritesByValue[value]);
                            changeItemTextureOpacity(matrix[r, c], cellOpacitiesByValue[value]);
                            changeItemTextColor(matrix[r, c], textColorsByValue[value]);
                        }
                        else
                        {
                            Debug.Log("ERROR: The dictionary has not the key " + value.ToString() + "!");
                        }
                    }
                }

                // Update the current index
                matrixCurrentTimeShift++;
                if (matrixCurrentTimeShift >= matrixSequenceLength)
                {
                    matrixCurrentTimeShift = 0;
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
            if (matrixCurrentTimeShift == 0)
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
                resetMatrix();

                // Request MEDUSA to process the trial
                state = STATE_WAITING_SELECTION;
                ServerMessage sm = new ServerMessage("processPlease");
                tcpClient.SendMessage(sm.ToJson());
            }
            else
            {
                // Make the flashings
                for (int r = 0; r < matrix.GetLength(0); r++)
                {
                    for (int c = 0; c < matrix.GetLength(1); c++)
                    {
                        int value = matrixItemSequence[r][c][matrixCurrentTimeShift];
                        if (cellSpritesByValue.ContainsKey(value))
                        {
                            changeItemTexture(matrix[r, c], cellSpritesByValue[value]);
                            changeItemTextureOpacity(matrix[r,c], cellOpacitiesByValue[value]);
                            changeItemTextColor(matrix[r,c], textColorsByValue[value]);
                        }
                        else
                        {
                            Debug.Log("ERROR: The dictionary has not the key " + value.ToString() + "!");
                        }
                    }
                }

                // Update the current index
                matrixCurrentTimeShift++;
                if (matrixCurrentTimeShift >= matrixSequenceLength)
                {
                    matrixCurrentTimeShift = 0;
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

/* ------------------------------------------- RASTER LATENCIES UTILS ------------------------------------------- */
public bool windowMoved()
    {
        // Get the size and position of the window using native DLLs
        Rect windowRect = new Rect();
        GetWindowRect(FindWindow(null, Application.productName), ref windowRect);

        // Update the info if the window has been moved
        if (((int)windowRect.Left != lastWindowLeft) || ((int)windowRect.Top != lastWindowTop))
        {
            lastWindowLeft = windowRect.Left;
            lastWindowTop = windowRect.Top;
            return true;
        }
        return false;
    }

    // This function converts a x,y measured positions inside a window into global pixels in function of the monitor resolution
    public int[] estimatePixelPosition(int x, int y)
    {
        // Get the size and position of the window using native DLLs
        Rect windowRect = new Rect();
        GetWindowRect(FindWindow(null, Application.productName), ref windowRect);

        // Map the pixed positions x, y inside the window to the global monitor pixels (multiple screens supported)
        int px_left = (windowRect.Left + x + 8) % Screen.currentResolution.width;
        int header = windowRect.Bottom - windowRect.Top - Screen.height;
        int px_top = (windowRect.Top + y + 8 + header) % Screen.currentResolution.height;
        return new int[] { px_left, px_top };
    }

    public struct Rect
    {
        public int Left { get; set; }
        public int Top { get; set; }
        public int Right { get; set; }
        public int Bottom { get; set; }
    }

}