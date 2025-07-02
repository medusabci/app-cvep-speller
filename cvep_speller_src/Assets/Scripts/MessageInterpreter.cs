// MEDUSA-PLATFORM 
// v2022.0 CHAOS
// www.medusabci.com

// MessageInterpreter for the c-VEP Speller (Unity app)
//      > Author: Víctor Martínez-Cagigal
//      > Date: 19/05/2022

// Versions:
//      - v1.0 (19/05/2022):    Initial message interpreter


using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using Newtonsoft.Json;

public class MessageInterpreter
{
    // This class provides a framework to decode all types of messages that are sent by MEDUSA-PLATFORM
    public MessageInterpreter()
    { 
    }

    /* ----------------------------------- DECODING FUNCTIONS ------------------------------------ */
    public string decodeEventType(string message) 
    {
        return EventTypeDecoder.getEventTypeFromJSON(message);
    }

    public ParameterDecoder decodeParameters(string message)
    {
        return ParameterDecoder.getParametersFromJSON(message);
    }

    public string decodeException(string message)
    {
        return ExceptionDecoder.getExceptionFromJSON(message);
    }

    public int[] decodeSelection(string message)
    {
        return SelectionDecoder.getSelectionFromJSON(message);
    }

    /* ----------------------------------- DECODING CLASSES ------------------------------------ */
    /** Class to decode the event_type first. */
    public class EventTypeDecoder
    {
        public string event_type;

        public static string getEventTypeFromJSON(string jsonString)
        {
            EventTypeDecoder event_type = JsonUtility.FromJson<EventTypeDecoder>(jsonString);
            return event_type.event_type;
        }
    }

    /** Class to decode parameters from the c-VEP Speller app (v1.0)
     * Check out the associated app_controller.py.
     **/
    public class ParameterDecoder
    {
        // Matrices
        public List<Matrix> matrices { get; set; }

        // RunSettings
        public string mode;
        public bool photodiodeEnabled;
        public int trainCycles;
        public int trainTrials;
        public int testCycles;
        public float fpsResolution;

        // Timings
        public float tPrevText;
        public float tPrevIddle;
        public float tFinishText;

        // Colors
        public string color_background;
        public string color_target_box;
        public string color_highlight_result_box;
        public string color_result_info_box;
        public string color_result_info_label;
        public string color_result_info_text;
        public string color_fps_good;
        public string color_fps_bad;
        public string color_box_0;
        public string color_box_1;
        public string color_text_0;
        public string color_text_1;

        //Opacity
        public float color_op_box_0;
        public float color_op_box_1;
        public float color_op_text_0;
        public float color_op_text_1;

        //Scenario
        public string scenario_name;
        public string scenario_path;

        public static ParameterDecoder getParametersFromJSON(string jsonString)
        {
            ParameterDecoder p = JsonConvert.DeserializeObject<ParameterDecoder>(jsonString);
            return p;
        }

        public class Matrix
        {
            public int n_row { get; set; }
            public int n_col { get; set; }
            public List<Target> item_list { get; set; }
        }

        public class Target
        {
            public int row { get; set; }
            public int col { get; set; }
            public string text { get; set; }
            public string label { get; set; }
            public int[] sequence { get; set; }
        }
    }

    public class ExceptionDecoder
    {
        public string exception;

        public static string getExceptionFromJSON(string jsonString)
        {
            ExceptionDecoder exception = JsonUtility.FromJson<ExceptionDecoder>(jsonString);
            return exception.exception;
        }
    }

    public class SelectionDecoder
    {
        public int[] selection_coords;

        public static int[] getSelectionFromJSON(string jsonString)
        {
            SelectionDecoder s = JsonUtility.FromJson<SelectionDecoder>(jsonString);
            return s.selection_coords;
        }
    }

    // Utility
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
}

public class ServerMessage
{
    public Dictionary<string, object> message = new Dictionary<string, object>();

    public ServerMessage(string action)
    {
        message.Add("event_type", action);
    }

    public void addValue(string key, object value)
    {
        message.Add(key, value);
    }

    public string ToJson()
    {
        return JsonConvert.SerializeObject(message);
    }
}
