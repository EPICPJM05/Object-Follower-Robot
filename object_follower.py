# -- coding: utf-8 --
import cv2
import numpy as np
import time
import multiprocessing as mp
import RPi.GPIO as GPIO

# ==================================================
# ---------------- MOTOR PROCESS -------------------
# ==================================================
def motor_process(command_queue, stop_event):

    GPIO.setmode(GPIO.BCM)

    IN1, IN2, IN3, IN4 = 17, 18, 27, 22
    GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)

    ENA, ENB = 12, 13
    GPIO.setup([ENA, ENB], GPIO.OUT)

    # ---- PWM SETUP ----
    pwm_A = GPIO.PWM(ENA, 1000)
    pwm_B = GPIO.PWM(ENB, 1000)
    pwm_A.start(0)
    pwm_B.start(0)

    # ---- MOTOR CALIBRATION (ADJUST IF NEEDED) ----
    LEFT_GAIN = 1.00     # Reduce if robot drifts right
    RIGHT_GAIN = 1.00

    # ---- STATE DEFINITIONS ----
    STATE_IDLE = 0
    STATE_MOVING = 1
    STATE_BRAKE = 2

    motor_state = STATE_IDLE
    motor_start_time = 0
    motor_duration = 0

    BRAKE_TIME = 0.070

    def stop_all():
        pwm_A.ChangeDutyCycle(0)
        pwm_B.ChangeDutyCycle(0)
        GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)

    def apply_brake():
        GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)

    def set_speed(duty):
        duty = max(30, min(100, duty))
        left_duty = max(0, min(100, duty * LEFT_GAIN))
        right_duty = max(0, min(100, duty * RIGHT_GAIN))

        pwm_A.ChangeDutyCycle(left_duty)
        pwm_B.ChangeDutyCycle(right_duty)

    def move(direction):
        if direction == "RIGHT":
            GPIO.output(IN1, 0); GPIO.output(IN2, 1)
            GPIO.output(IN3, 1); GPIO.output(IN4, 0)
        elif direction == "LEFT":
            GPIO.output(IN1, 1); GPIO.output(IN2, 0)
            GPIO.output(IN3, 0); GPIO.output(IN4, 1)
        elif direction == "FORWARD":
            GPIO.output(IN1, 1); GPIO.output(IN2, 0)
            GPIO.output(IN3, 1); GPIO.output(IN4, 0)
        elif direction == "BACKWARD":
            GPIO.output(IN1, 0); GPIO.output(IN2, 1)
            GPIO.output(IN3, 0); GPIO.output(IN4, 1)

    try:
        while not stop_event.is_set():
            now = time.monotonic()

            if motor_state == STATE_IDLE and not command_queue.empty():
                direction, duration, speed = command_queue.get()
                set_speed(speed)
                move(direction)
                motor_start_time = now
                motor_duration = duration
                motor_state = STATE_MOVING

            if motor_state == STATE_MOVING:
                if now - motor_start_time >= motor_duration:
                    apply_brake()
                    motor_start_time = now
                    motor_state = STATE_BRAKE

            elif motor_state == STATE_BRAKE:
                if now - motor_start_time >= BRAKE_TIME:
                    stop_all()
                    motor_state = STATE_IDLE

    finally:
        stop_all()
        pwm_A.stop()
        pwm_B.stop()
        GPIO.cleanup()


# ==================================================
# ---------------- VISION PROCESS ------------------
# ==================================================
def vision_process(command_queue, stop_event):

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Camera not opened")
        return

    MIN_AREA = 100
    X_DEAD = 90
    Y_DEAD = 90

    BASE_DURATION = 0.015
    KP_TURN = 0.12
    KP_MOVE = 0.12
    MAX_DURATION = 0.18

    LOWER_BLUE = (98,120,90)
    UPPER_BLUE = (120, 255, 255)

    print("Blue object tracking + dynamic speed control started")
    prev_time = time.monotonic()
    fps = 0

    
    try:
        while not stop_event.is_set():
            
            current_time = time.monotonic()
            dt = current_time - prev_time
            prev_time = current_time

            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  # exponential smoothing


            ret, frame = cap.read()
            if not ret or frame is None:
                continue

	    # Rotate 90 degrees clockwise
	    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

	    h, w = frame.shape[:2]
            FRAME_CX = w // 2
            FRAME_CY = h // 2

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            action = "STOP"

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)

                if area > MIN_AREA:

                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue

                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    if 0.5 < circularity <= 1.2:

                        x, y, w_box, h_box = cv2.boundingRect(cnt)
                        cx = int(x + w_box / 2)
                        cy = int(y + h_box / 2)

                        error_x = cx - FRAME_CX
                        error_y = cy - FRAME_CY

                        error_norm_x = error_x / FRAME_CX
                        error_norm_y = error_y / FRAME_CY

                        if abs(error_x) <= X_DEAD and abs(error_y) <= Y_DEAD:
                            action = "STOP"

                        elif abs(error_x) > abs(error_y):

                            duration = BASE_DURATION + KP_TURN * abs(error_norm_x)
                            duration = min(duration, MAX_DURATION)
                            speed = 40 + 60 * abs(error_norm_x)

                            if error_x < 0:
                                if command_queue.empty():
                                    command_queue.put(("RIGHT", duration, speed))
                                action = f"RIGHT {duration:.3f}s"
                            else:
                                if command_queue.empty():
                                    command_queue.put(("LEFT", duration, speed))
                                action = f"LEFT {duration:.3f}s"

                        else:

                            duration = BASE_DURATION + KP_MOVE * abs(error_norm_y)
                            duration = min(duration, MAX_DURATION)
                            speed = 40 + 60 * abs(error_norm_y)

                            if error_y < 0:
                                if command_queue.empty():
                                    command_queue.put(("FORWARD", duration, speed))
                                action = f"FORWARD {duration:.3f}s"
                            else:
                                if command_queue.empty():
                                     command_queue.put(("BACKWARD", duration, speed))
                                action = f"BACKWARD {duration:.3f}s"

                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            else:
                action = "NO OBJECT"

            cv2.line(frame, (FRAME_CX, 0), (FRAME_CX, h-1), (255, 0, 0), 1)
            cv2.line(frame, (0, FRAME_CY), (w-1, FRAME_CY), (255, 0, 0), 1)

            cv2.putText(frame, action, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)


            cv2.imshow("Smart Dustbin Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
    		stop_event.set()
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ==================================================
# -------------------- MAIN ------------------------
# ==================================================
if __name__ == "__main__":

    mp.set_start_method("spawn")

    command_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    motor_p = mp.Process(target=motor_process, args=(command_queue, stop_event))
    vision_p = mp.Process(target=vision_process, args=(command_queue, stop_event))

    motor_p.start()
    vision_p.start()

    vision_p.join()
    stop_event.set()
    motor_p.join()
