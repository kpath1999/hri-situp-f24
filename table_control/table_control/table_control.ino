// #include <Arduino_FreeRTOS.h>

unsigned int act_time = 750;
unsigned int up_mos = 3;
unsigned int down_mos = 2;
String command;

// // define two tasks for Blink & AnalogRead
// void TaskTableSpeed( void *pvParameters );
// void TaskSerial( void *pvParameters );

// the setup function runs once when you press reset or power the board
void setup() {

  pinMode(up_mos, OUTPUT);
  pinMode(down_mos, OUTPUT);
  
  // initialize serial communication at 9600 bits per second:
  Serial.begin(115200);
  Serial.println("Starting");
  
  while (!Serial) {
  }

  // xTaskCreate(
  //   TaskTableSpeed
  //   ,  "TaskTableSpeed"   // A name just for humans
  //   ,  128  // This stack size can be checked & adjusted by reading the Stack Highwater
  //   ,  NULL
  //   ,  3  // Priority, with 3 (configMAX_PRIORITIES - 1) being the highest, and 0 being the lowest.
  //   ,  NULL );

  // xTaskCreate(
  //   TaskSerial
  //   ,  "TaskSerial"
  //   ,  128  // Stack size
  //   ,  NULL
  //   ,  1  // Priority
  //   ,  NULL );

  // Now the task scheduler, which takes over control of scheduling individual tasks, is automatically started.
}

void loop()
{
  // Serial.println("Table go up");
  // digitalWrite(down_mos, LOW);
  // digitalWrite(up_mos, HIGH);
  // delay(1000);
  // digitalWrite(up_mos, LOW);

  // delay(2000);
  // Empty. Things are done in Tasks.
  while(!Serial.available());

    command = Serial.readStringUntil('\n');
    command.trim();
    Serial.println(command);

    if(command == "up")
    {
      Serial.println("Table go up");
      digitalWrite(down_mos, LOW);
      digitalWrite(up_mos, HIGH);
      delay(act_time);
      digitalWrite(up_mos, LOW);
    }

    else if(command == "down")
    {
      Serial.println("Table go down");
      digitalWrite(up_mos, LOW);
      digitalWrite(down_mos, HIGH);
      delay(act_time);
      digitalWrite(down_mos, LOW);
    }
    else{
      act_time = command.toInt();
    }
}

/*--------------------------------------------------*/
/*---------------------- Tasks ---------------------*/
/*--------------------------------------------------*/

// void TaskTableSpeed(void *pvParameters)  // This is a task.
// {
//   (void) pvParameters;
//   float speed;
//   pinMode(speed_mos, OUTPUT);

//   for (;;) // A Task shall never return or exit.
//   {
//     digitalWrite(speed_mos, HIGH);
//     vTaskDelay( speed*10 / portTICK_PERIOD_MS );
//     digitalWrite(speed_mos, LOW);
//     vTaskDelay( (1-speed)*10 / portTICK_PERIOD_MS);

//   }
// }

// void TaskSerial(void *pvParameters)  // This is a task.
// {
//   (void) pvParameters;
//   Serial.println("Waiting");
//   for (;;)
//   {

    

//   }
// }
