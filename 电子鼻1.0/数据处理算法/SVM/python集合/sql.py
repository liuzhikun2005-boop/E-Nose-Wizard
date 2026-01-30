import serial
import mysql.connector
import tkinter as tk


# Function to handle start button click
def start():
    global ser, mycursor, mydb

    # Configure serial port
    ser = serial.Serial('COM5', 115200)  # Modify the port and baud rate as per your setup

    # Configure MySQL connection
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="dir99",
        database="nose")
    mycursor = mydb.cursor()

    # Start reading and processing serial data
    read_serial_data()


# Function to handle stop button click
def stop():
    global ser, mycursor, mydb

    # Close serial port and database connection
    ser.close()
    mycursor.close()
    mydb.close()


# Function to read and process serial data
def read_serial_data():
    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode().strip()
                print("Received:", data)

                if data.startswith('DATA,'):
                    values = data.split(',')[1:]
                    if len(values) == 4:
                        sql = "INSERT INTO dianzibi (GM102B, GM302B, GM502B, GM702B, CLASS) VALUES (%s, %s, %s, %s, %s)"
                        class_value = 0
                        val = (values[0], values[1], values[2], values[3], class_value)
                        mycursor.execute(sql, val)
                        mydb.commit()
                        mycursor.execute("SELECT * FROM dianzibi")
                        print("Table dianzibi:")
                        for row in mycursor.fetchall():
                            print(row)

    except KeyboardInterrupt:
        print("Exiting program...")


# Create Tkinter window
window = tk.Tk()
window.title("Serial Data Logger")

# Start button
start_button = tk.Button(window, text="Start", command=start)
start_button.pack()

# Stop button
stop_button = tk.Button(window, text="Stop", command=stop)
stop_button.pack()

window.mainloop()
