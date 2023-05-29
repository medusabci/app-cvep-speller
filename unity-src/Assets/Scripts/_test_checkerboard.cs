using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;

public class CheckerTest : MonoBehaviour
{

    public Sprite[] checkerboards;
    public int spatial_cycles = 64;
    public int cellSize = 500;
    private GameObject mainTestCell;

    // Start is called before the first frame update
    void Start()
    {
        checkerboards = generateCheckerboards(cellSize, spatial_cycles, new Color(0f, 0f, 0f, 1f), new Color(1f, 1f, 1f, 1f));
        mainTestCell = GameObject.Find("Test_Cell_Main");
        mainTestCell.GetComponent<Image>().sprite = checkerboards[0];

        RectTransform rt = mainTestCell.GetComponent<RectTransform>();
        Debug.Log(rt.rect.width);
        Debug.Log(rt.rect.height);

        Debug.Log(checkerboards[0].rect.width);
        Debug.Log(checkerboards[0].rect.height);

        mainTestCell.GetComponent<RectTransform>().sizeDelta = new Vector2(checkerboards[0].rect.width, checkerboards[0].rect.height);
        Debug.Log(rt.rect.width);
        Debug.Log(rt.rect.height);
    }

    // Update is called once per frame
    void Update()
    {
        
    }


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

        int texturesize = 2 * spatialCycles * blockSize;
        Debug.Log("texture size: " + texturesize);
        Debug.Log("cell size: " + imageSize);

        // Ignore spatial pixel interpolation
        texturePositive.wrapMode = TextureWrapMode.Mirror;
        texturePositive.filterMode = FilterMode.Point;
        textureNegative.wrapMode = TextureWrapMode.Mirror;
        textureNegative.filterMode = FilterMode.Point;

        // Create the sprites
        Sprite spritePositive = Sprite.Create(texturePositive, new Rect(0, 0, texturePositive.width, texturePositive.height), Vector2.one * 0.5f);
        Sprite spriteNegative = Sprite.Create(textureNegative, new Rect(0, 0, textureNegative.width, textureNegative.height), Vector2.one * 0.5f);
        return new Sprite[] { spritePositive, spriteNegative };
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
