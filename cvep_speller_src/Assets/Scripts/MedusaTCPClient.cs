// MEDUSA-PLATFORM 
// v2022.0 CHAOS
// www.medusabci.com

// MedusaTCPClient
//      > Author: Víctor Martínez-Cagigal

// Versions:
//      - v1.0 (29/09/2021):    Support for MEDUSA 2.0 server using asyncio
//      - v2.0 (24/03/2022):    Updated so asyncio is not used anymore
//      - v2.1 (31/03/2022):    Now the client can decode merged messages
//		- v2.2 (14/09/2022):	Minor update in function isConnected()

using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

public class MedusaTCPClient
{
	// Required instances
	public TcpClient socketConnection;
	private Thread clientReceiveThread;
	private Manager callback;

	// Server IP and port
	// TODO: TAKE THE IP and port from MEDUSA as parameters -exe
	private IPAddress IP;
	private int port;

	// Message reading
	public string recvBuffer = "";
	private int PROTOHEADER_LEN = 2;
	private byte[] _recvBuffer = new byte[0];
	private int _jsonHeaderLen = -1;
	private Header _jsonHeader = null;
	private string _recvMessage = null;


	/// <summary>
	/// Constructor of the MedusaTCPClient. It provides an asynchronous client that connects to MEDUSA, which is running
	/// a server that sends and receives parameters to and from this client.
	/// </summary>
	/// <param name="manager"> A Manager instance is required as a callback to receive the data. </param>
	/// <param name="serverIP"> IPAddress instance of the server's IP. </param>
	/// <param name="serverPort"> Integer that represents the server's port. </param>
	public MedusaTCPClient(Manager manager, IPAddress serverIP, int serverPort)
	{
		callback = manager;
		this.IP = serverIP;
		this.port = serverPort;
	}

	/// <summary>
	/// This method checks if the client is being connected to the server.
	/// </summary>
	/// <returns>Boolean: true is connected, false otherwise. </returns>
	public bool isConnected()
	{
		if (socketConnection != null && socketConnection.Connected)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	/// <summary>
	/// This method starts the client, which basically creates a thread for listening and initializes the connection.
	/// </summary>
	public void Start()
	{
		try
		{
			clientReceiveThread = new Thread(new ThreadStart(ListenForData));
			clientReceiveThread.IsBackground = true;
			clientReceiveThread.Start();
		}
		catch (Exception e)
		{
			Debug.Log("On client connect exception " + e);
		}
	}

	/// <summary>
	/// This method merges two byte[] arrays
	/// </summary>
	public static byte[] addBytes(byte[] first, byte[] second)
	{
		byte[] bytes = new byte[first.Length + second.Length];
		Buffer.BlockCopy(first, 0, bytes, 0, first.Length);
		Buffer.BlockCopy(second, 0, bytes, first.Length, second.Length);
		return bytes;
	}

	/// <summary>
	/// This method creates a connection and listens forever to the server in order to check is data is received. If so, 
    /// the method decodes the message: first, the proto-header (which determines the length of the next header); second,
    /// the JSON header (which determines the type of encoding and the lenght of the message); and third, the JSON message.
    /// Then, this function returns the processed messages to the Manager instance (i.e., callback).
	/// </summary>
	private void ListenForData()
	{
		try
		{
			socketConnection = new TcpClient(this.IP.ToString(), this.port);
			Byte[] bytes = new Byte[4096];
			Debug.Log("Client listening at " + this.IP.ToString() + ":" + this.port);
			while (true)
			{
				// Get a stream object for reading 				
				using (NetworkStream stream = socketConnection.GetStream())
				{
					int length;
					// Read incomming stream into byte arrary. 					
					while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
					{
						// Append binary data into the receiving buffer
						byte[] incomingData = new byte[length];
						Array.Copy(bytes, 0, incomingData, 0, length);
						this._recvBuffer = addBytes(this._recvBuffer, incomingData);

						// While there is something to read
						while (this._recvBuffer.Length > 0)
						{
							// Decoding first protoheader 
							if (this._jsonHeaderLen == -1)
							{
								if (this._recvBuffer.Length >= this.PROTOHEADER_LEN)
								{
									byte[] protoHeaderData = this._recvBuffer[..2];
									Array.Reverse(protoHeaderData);                     // The protoheader is encoded in big-endian (network)
									this._jsonHeaderLen = BitConverter.ToInt16(protoHeaderData, 0);
									this._recvBuffer = this._recvBuffer[2..];
								} else
								{
									break;
								}
							}
							// Decoding the JSON header
							if (this._jsonHeaderLen != -1)
							{
								if (this._jsonHeader == null)
								{
									if (this._recvBuffer.Length >= this._jsonHeaderLen)
									{
										string jsonHeaderStr = Encoding.UTF8.GetString(this._recvBuffer[..this._jsonHeaderLen]);
										// Workaround to make the header de-serializable (removing all "-")
										jsonHeaderStr = jsonHeaderStr.Replace("content-type", "content_type");
										jsonHeaderStr = jsonHeaderStr.Replace("content-encoding", "content_encoding");
										jsonHeaderStr = jsonHeaderStr.Replace("content-length", "content_length");
										// Deserialize to custom object
										this._jsonHeader = JsonConvert.DeserializeObject<Header>(jsonHeaderStr);
										this._recvBuffer = this._recvBuffer[this._jsonHeaderLen..];
									} else
									{
										break;
									}
								}
							}
							// Decode message
							if (this._jsonHeader != null)
							{
								if (this._recvMessage == null)
								{
									if (this._recvBuffer.Length >= (int)this._jsonHeader.content_length)
									{
										switch (this._jsonHeader.content_encoding)
										{
											case "utf-8":
												this._recvMessage = Encoding.UTF8.GetString(this._recvBuffer[..this._jsonHeader.content_length]);
												break;
											case "utf-7":
												this._recvMessage = Encoding.UTF7.GetString(this._recvBuffer[..this._jsonHeader.content_length]);
												break;
											case "utf-32":
												this._recvMessage = Encoding.UTF32.GetString(this._recvBuffer[..this._jsonHeader.content_length]);
												break;
											case "unicode":
												this._recvMessage = Encoding.Unicode.GetString(this._recvBuffer[..this._jsonHeader.content_length]);
												break;
											case "ascii":
												this._recvMessage = Encoding.ASCII.GetString(this._recvBuffer[..this._jsonHeader.content_length]);
												break;
										}
										this._recvBuffer = this._recvBuffer[this._jsonHeader.content_length..];
									} else
									{
										break;
									}
								}
							}
							// Interpret message
							if (this._recvMessage != null)
							{
								callback.interpretMessage(this._recvMessage);
								this._jsonHeaderLen = -1;
								this._jsonHeader = null;
								this._recvMessage = null;
							}
						}

					}
				}
			}
		}
		catch (SocketException socketException)
		{
			Debug.Log("Socket exception: " + socketException);
			socketConnection.Close();
			socketConnection = null;
			callback.quitApplicationFromException();
		}
		catch (System.IO.IOException ioException)
		{
			Debug.Log("IOException exception: " + ioException);
			socketConnection.Close();
			socketConnection = null;
			callback.quitApplicationFromException();
		}
	}

	/// <summary>
	/// This method sends asynchronously any message to the server, provided the connection has been previously established.
	/// </summary>
	/// <param name="clientMessage"> String that contains the message to send. </param>
	public void SendMessage(string clientMessage)
	{
		if (!this.isConnected())
		{
			return;
		}
		try
		{
			// Get a stream object for writing. 			
			NetworkStream stream = socketConnection.GetStream();
			if (stream.CanWrite)
			{
				// Encode message
				byte[] bClientMessage = Encoding.UTF8.GetBytes(clientMessage);

				// Compute the headers
				Dictionary<string, object> msgJSONHeader = new Dictionary<string, object>();
				msgJSONHeader["byteorder"] = "little";
				msgJSONHeader["content-type"] = "text/json";
				msgJSONHeader["content-encoding"] = "utf-8";
				msgJSONHeader["content-length"] = bClientMessage.Length;
				byte[] bClientMessageJSONHeader = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(msgJSONHeader));

				// Compute the protoheader
				Int16 hl = Convert.ToInt16(bClientMessageJSONHeader.Length);
				byte[] bClientMessageProtoHeader = BitConverter.GetBytes(hl);
				Array.Reverse(bClientMessageProtoHeader, 0, bClientMessageProtoHeader.Length);	// The protoheader is encoded in LITTLE ENDIAN (network)

				// Concatenate everything
				byte[] header = addBytes(bClientMessageProtoHeader, bClientMessageJSONHeader);
				byte[] message = addBytes(header, bClientMessage);

				// Write byte array to socketConnection stream.
				stream.Write(message, 0, message.Length);
				Debug.Log("Client sent: " + clientMessage);
			}
		}
		catch (SocketException socketException)
		{
			Debug.Log("Socket exception: " + socketException);
		}
	}

	/// <summary>
	/// This method stops the MedusaTCPClient, killing the listening thread and closing the connection.
	/// </summary>
	public bool Stop()
	{
		if (socketConnection != null)
		{
			socketConnection.Close();
			socketConnection = null;
		}
		clientReceiveThread.Abort();
		Debug.Log("Client closed");
		return true;
	}
}

public class Header
{
	public string byteorder { get; set; }
	public string content_type { get; set; }
	public string content_encoding { get; set; }
	public int content_length { get; set; }
}