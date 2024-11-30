#include <Arduino.h>
#include <math.h>


// 定数
const double VMAX = 12.0;
const int16_t DUTY_MAX = 2047;
const int16_t DUTY_LIM = 2040;
const int COUNT_NUM = 4000;
const double dt = 1e-03;
const double resolution = 2.0 * PI / 2400.0;

// ゲイン定数
const double K0 = -6.376e+02;
const double K1 = -2.733e+01;
const double G = 5.477e+03;
const double Lp0 = -1.774e+03;    // 極配置法
const double Lp1 = -4.046e+05;    // 極配置法
const double Lk0 = -1.001;           // 定常カルマンフィルタ
const double Lk1 = -7.751e-01;       // 定常カルマンフィルタ

// ピン定義
const uint8_t Led1 = 27, Led2 = 14;
const uint8_t encoder1_A = 25, encoder1_B = 26;
const uint8_t M1A = A12, M1B = A10;
const uint8_t Switch1 = 33, Switch2 = 32;

// グローバル変数
volatile int enc_count;
double vout = 0.0;
double theta[2] = {0.0, 0.0}, dtheta[2] = {0.0, 0.0};
double theta_hat[2] = {0.0, 0.0}, dtheta_hat[2] = {0.0, 0.0};
double target = PI * 0.5;
bool flag_button_pushed = false, flag_debug = false;
bool flag_is_discrete = true;  // true:定常カルマンフィルタ、false:極配置法
uint32_t cnt_debug = 0, i = 0;

// デバッグデータ
double theta_debug[COUNT_NUM + 1], input_debug[COUNT_NUM + 1];

// タイマー
hw_timer_t *timer = NULL, *timer1 = NULL;

// 関数プロトタイプ
int16_t state_feedback(double x, double dx, double x_ref);
void state_observer(double y, double u, bool is_discrete_flag);
int16_t Msequence_signal_output(int m_sequence);
void IRAM_ATTR control();
void IRAM_ATTR LED_Flashing();
void IRAM_ATTR button1_pushed();
void IRAM_ATTR button2_pushed();
void encoder1_pulse_a();
void encoder1_pulse_b();

void setup() 
{
  // シリアル通信設定
  Serial.begin(921600);

  // GPIOおよびPWM設定
  pinMode(Led1, OUTPUT);
  pinMode(Led2, OUTPUT);
  pinMode(Switch1, INPUT_PULLUP);
  pinMode(Switch2, INPUT_PULLUP);
  pinMode(encoder1_A, INPUT_PULLUP);
  pinMode(encoder1_B, INPUT_PULLUP);

  // PWM設定
  ledcSetup(1, 20000, 11);
  ledcSetup(2, 20000, 11);
  ledcAttachPin(M1A, 1);
  ledcAttachPin(M1B, 2);

  // 割り込み設定
  attachInterrupt(encoder1_A, encoder1_pulse_a, CHANGE);
  attachInterrupt(encoder1_B, encoder1_pulse_b, CHANGE);
  attachInterrupt(Switch1, button1_pushed, FALLING);
  attachInterrupt(Switch2, button2_pushed, FALLING);

  // タイマー設定
  timer = timerBegin(0, 80, true);
  timerAttachInterrupt(timer, &LED_Flashing, true);
  timerAlarmWrite(timer, 75000, true);
  timerAlarmEnable(timer);

  timer1 = timerBegin(1, 80, true);
  timerAttachInterrupt(timer1, &control, true);
  timerAlarmWrite(timer1, 1000, true);

  Serial.println("ESP32 Initialized");
  digitalWrite(Led2, HIGH);
  delay(1000);
  timerAlarmEnable(timer1);
}

void loop() 
{
  if (flag_debug) 
  {
    if (i < COUNT_NUM) 
    {
      Serial.printf("%u, %.3f, %.6f\r\n", i, input_debug[i], theta_debug[i]);
      i++;
    }
  }
  Serial.printf("%lu, %.2f, %.4f, %.4f, %.4f\r\n",
                millis(), vout, theta[0], theta_hat[0], dtheta_hat[0]);
  delay(3);
}

// 状態フィードバック制御
int16_t state_feedback(double x, double dx, double x_ref) 
{
  static double integral_error = 0.0;
  double error = x_ref - x;
  integral_error = constrain(integral_error + error * dt, -VMAX, VMAX);

  double u = K0 * x + K1 * dx + G * integral_error;
  u = constrain(u, -VMAX, VMAX);

  int16_t duty = (int16_t)(u * DUTY_MAX / VMAX);
  duty = constrain(duty, -DUTY_LIM, DUTY_LIM);

  vout = u;
  return isfinite(vout) ? duty : 0;
}

// 状態オブザーバ
void state_observer(double y, double u, bool is_discrete_flag) 
{
  if (is_discrete_flag) 
  {
    theta_hat[0] = (theta_hat[1] - y) * Lk0 + 1.953e-5 * u + theta_hat[1] + 0.000987 * dtheta_hat[1];
    dtheta_hat[0] = (theta_hat[1] - y) * Lk1 + 0.0389 * u + 0.9747 * dtheta_hat[1];
  } 
  else 
  {
    theta_hat[0] += ((theta_hat[1] - y) * Lp0 + dtheta_hat[1]) * dt;
    dtheta_hat[0] += ((theta_hat[1] - y) * Lp1 + 39.4 * u - 25.6 * dtheta_hat[1]) * dt;
  }
  theta_hat[1] = theta_hat[0];
  dtheta_hat[1] = dtheta_hat[0];
}

// 割り込み処理
void IRAM_ATTR LED_Flashing() 
{
  digitalWrite(Led2, !digitalRead(Led2));
}

void IRAM_ATTR control() 
{
  theta[0] = enc_count * resolution;
  dtheta[0] = (theta[0] - theta[1]) / dt;

  if (flag_button_pushed) 
  {
    state_observer(theta[0], vout, flag_is_discrete);
    int16_t duty = state_feedback(theta_hat[0], dtheta_hat[0], target);
    if (cnt_debug < COUNT_NUM) 
    {
      input_debug[cnt_debug] = vout;
      theta_debug[cnt_debug] = theta[0];
      cnt_debug++;
    }
    ledcWrite(1, duty > 0 ? 0 : abs(duty));
    ledcWrite(2, duty > 0 ? abs(duty) : 0);
  }

  theta[1] = theta[0];
  dtheta[1] = dtheta[0];
}

void IRAM_ATTR button1_pushed() 
{
  flag_button_pushed = !flag_button_pushed;
  digitalWrite(Led1, flag_button_pushed ? HIGH : LOW);
}

void IRAM_ATTR button2_pushed() 
{
  flag_debug = !flag_debug;
}

void encoder1_pulse_a()
{
  if (digitalRead(encoder1_A) == 1)
  {
    if (digitalRead(encoder1_B) == 0)
      enc_count++;
    else
      enc_count--;
  }else
  {
    if (digitalRead(encoder1_B) == 1)
      enc_count++;
    else
      enc_count--;
  } 
}
void encoder1_pulse_b()
{
  if (digitalRead(encoder1_B) == 1)
  {
    if (digitalRead(encoder1_A) == 1)
      enc_count++;
    else
      enc_count--;
  }else
  {
    if (digitalRead(encoder1_A) == 0)
      enc_count++;
    else
      enc_count--;
  }
}