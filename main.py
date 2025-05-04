#main
import cv2
import numpy as np
from ultralytics import YOLO
from Minimax import MinimaxChessAI
from TTS import TTS 
import lmstudio as lms
from STT import STT

# Starting the models
model = YOLO(r"C:\Users\manue\OneDrive\Documentos\Universidad\Proyecto\código\yolotry3.pt")
chess_ai = MinimaxChessAI()
tts = TTS() 
stt =STT(language="en", record_seconds=10)#Change for any case

piece_positions={}

class_names = ['b-bishop', 'b-king', 'b-knight', 'b-pawn', 'b-queen', 'b-rook', 'w-bishop', 'w-king', 'w-knight', 'w-pawn', 'w-queen', 'w-rook']

#variables temp filter chess corners
prev_points = None
alpha = 0.2

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized

def get_inner_contour(contour, margin_px=20):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    center = np.mean(approx, axis=0)[0]
    inner_contour = []
    for point in approx:
        vector = point[0] - center
        unit_vector = vector / np.linalg.norm(vector)
        new_point = point[0] - unit_vector * margin_px
        inner_contour.append([new_point])
    
    return np.array(inner_contour, dtype=np.int32)

def order_points(points):
    #order points, adjust if it is need, but normally a1->3
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[1] = points[np.argmin(s)]
    rect[3] = points[np.argmax(s)]
    
    diff = np.diff(points, axis=1)
    rect[2] = points[np.argmin(diff)]
    rect[0] = points[np.argmax(diff)]
    return rect

def draw_chessboard_cells(warped_board, detected_pieces=None):
    square_size = warped_board.shape[0] // 8 
    board_with_pieces = warped_board.copy()
    
    piece_positions = {}  
    
    for row in range(8):
        for col in range(8):
            x1, y1 = col * square_size, row * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            
            cv2.rectangle(board_with_pieces, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            notation = f"{chr(97 + col)}{8 - row}"  
            cv2.putText(board_with_pieces, notation, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # pieces detected draw them on proper boxes
    if detected_pieces:
        for piece in detected_pieces:
            x, y, w, h, piece_class, conf = piece
            # determine box for piece
            center_x = x + w/2
            center_y = y + h/2
            col = int(center_x // square_size)
            row = int(center_y // square_size)
            
            if 0 <= col < 8 and 0 <= row < 8:  
                square_notation = f"{chr(97 + col)}{8 - row}"
            
                top_left = (int(x), int(y))
                bottom_right = (int(x + w), int(y + h))
                cv2.rectangle(board_with_pieces, top_left, bottom_right, (0, 255, 0), 3)
                
                piece_label = class_names[piece_class]
                cv2.putText(board_with_pieces, piece_label, 
                           (top_left[0] + 5, top_left[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                piece_positions[square_notation] = piece_label
    
    # show info detected pieces
    if detected_pieces:
        y_offset = 20
        cv2.putText(board_with_pieces, "Detected pieces:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        for square, piece in piece_positions.items():
            cv2.putText(board_with_pieces, f"{square}: {piece}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    return board_with_pieces, piece_positions


def explain_term(term):
    model = lms.llm("hermes-3-llama-3.2-3b")
    
    prompt = f"""
    You are Chess Coach, a friendly AI assistant designed to help chess beginners and amateurs improve their game. Your responses should always be concise, informative, and easy to understand.
    
    User Question Format:
    {term}
    The above is the user's chess question. Please respond to this specific question using the guidelines below.
    
    Response Guidelines:

    Always answer in English, regardless of the language used in the question
    Keep all answers under 500 characters total (including spaces and punctuation)
    Be direct and get to the point quickly
    When discussing chess positions, use algebraic notation
    When recommending moves or discussing numbers, spell them out in words rather than using digits (e.g., "knight to f three" instead of "Nf3", "twenty-four" instead of "24")

    When responding to move recommendations:

    Provide a clear, specific answer with the exact move(s) to make
    Spell out the piece and destination (e.g., "Move your queen to d five")
    Briefly explain why this is a good move in one short sentence when possible
    Avoid listing multiple options unless specifically asked for alternatives

    When responding to non-chess questions:

    Politely explain that you are a chess assistant designed to help with chess-related questions
    Attempt to interpret how their question might relate to chess
    Offer to help with a chess-related topic instead

    Example responses:
    For move advice: "Move your knight to f three. This helps control the center and allows for kingside castling."
    For non-chess question: "I'm a chess assistant designed to help with chess questions. Your question about weather doesn't relate to chess, but I'd be happy to discuss chess openings or strategies instead."
    Always prioritize clarity and brevity in your responses while maintaining a helpful, encouraging tone for chess learners.
    """
    
    return model.respond(prompt)



while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = preprocess_image(frame)
    edges = cv2.Canny(processed, 150, 200)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    
    output = frame.copy()
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        inner_contour = get_inner_contour(largest_contour, margin_px=33)
        
        cv2.drawContours(output, [inner_contour], -1, (255, 0, 0), 2)
        
        epsilon = 0.1 * cv2.arcLength(inner_contour, True)
        approx = cv2.approxPolyDP(inner_contour, epsilon, True) 
        
        if len(approx) == 4:
            points = np.array([point[0] for point in approx], dtype="float32")
            current_points = order_points(points)
            
            if prev_points is None:
                prev_points = current_points
            else:
                ordered_points = prev_points * (1 - alpha) + current_points * alpha
                prev_points = ordered_points
            
            for i, (x, y) in enumerate(prev_points):
                cv2.circle(output, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(output, str(i), (int(x), int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            width, height = 600, 600
            dst = np.array([[0, 0], [width-1, 0], 
                           [width-1, height-1], [0, height-1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(prev_points, dst)
            warped = cv2.warpPerspective(frame, M, (width, height))
            
            #use the model after the transform of the perspective
            detected_pieces_warped = []
            results = model(warped, stream=True)
            
            for result in results:
                boxes = result.boxes.data.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2, conf, class_id = box
                    if conf > 0.5:  # trust filter
                        class_id = int(class_id)
                        w = x2 - x1
                        h = y2 - y1
                        
                        detected_pieces_warped.append((x1, y1, w, h, class_id, conf))
            
            warped_with_cells, piece_positions = draw_chessboard_cells(warped, detected_pieces_warped)
            
            # Show detections over img transformed
            warped_detection = warped.copy()
            for x1, y1, w, h, class_id, conf in detected_pieces_warped:
                cv2.rectangle(warped_detection, (int(x1), int(y1)), 
                             (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
                label = f"{class_names[class_id]} {conf:.2f}"
                cv2.putText(warped_detection, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Detections in transformed board", warped_detection)
            cv2.imshow("Board with detected pieces", warped_with_cells)
    
    cv2.imshow("Board detection", output)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('n'):
        if 'piece_positions' in locals() and piece_positions:
            print(f"DEBUG - Piece positions: {piece_positions}")
            
            ia_positions = {
                "white_king": None,
                "black_king": None,
                "black_rook": None
            }
            
            for square, piece in piece_positions.items():
                if len(square) == 2 and square[0].isalpha() and square[1].isdigit():
                    if piece == "w-king":
                        ia_positions["white_king"] = square
                    elif piece == "b-king":
                        ia_positions["black_king"] = square
                    elif piece == "b-rook":
                        ia_positions["black_rook"] = square
            
            if all(ia_positions.values()): 
                try:
                    origin, dest = chess_ai.get_best_move(ia_positions)
                    print(f"AI Movement: {origin} -> {dest}")
                    
                    tts.speak_move(origin, dest)
                    
                    # See movement on the board
                    cv2.putText(warped_with_cells, f"Move: {origin} -> {dest}", 
                               (10, warped_with_cells.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if origin and dest:
                        square_size = warped.shape[0] // 8
                        
                        source_col = ord(origin[0]) - ord('a')
                        source_row = 8 - int(origin[1])
                        dest_col = ord(dest[0]) - ord('a')
                        dest_row = 8 - int(dest[1])
                        
                        start_point = (source_col * square_size + square_size//2,
                                      source_row * square_size + square_size//2)
                        end_point = (dest_col * square_size + square_size//2,
                                    dest_row * square_size + square_size//2)
                        
                        cv2.arrowedLine(warped_with_cells, start_point, end_point, 
                                       (0, 0, 255), 2, tipLength=0.3)
                    
                    cv2.imshow("Board with AI move", warped_with_cells)
                    
                except Exception as e:
                    print(f"Error calculating the movement: {e}")
                    
                    tts.speak("Error calculating the movement")
            else:
                print("Error: Missing pieces on the board")
                missing = [k for k, v in ia_positions.items() if v is None]
                print(f"Missing pieces: {missing}")
                print(f"Pieces founded: {ia_positions}")
        else:
            print("The board has not been detected correctlye")
            
    elif key == ord('t'):
        print("\n" + "="*50)
        print("ASK YOUR DOUBTS")
        print("="*50)

        try:
            print("Recording...")
            term = stt.listen()

            if term and term.strip():
                print(f"Recognized term: '{term}'")
                print("Generating explanation...")

                explanation = explain_term(term)
                print(dir(explanation))

                # Response from the LLM
                if hasattr(explanation, 'text'):  
                    explanation = explanation.text
                elif hasattr(explanation, '__str__'):  
                    explanation = str(explanation)
                else:  
                    explanation = "Explanation can not be generated"

                print("\nEXPLANATION:")
                print("-"*50)
                print(explanation)
                print("-"*50)

                if explanation.strip():
                    tts.speak(explanation)
                else:
                    raise ValueError("There is not explanation found")

            else:
                print("No term was recognized")
                tts.speak("I did not understand the term. Please try again.")

        except Exception as e:
            print(f"Error processing chess term: {e}")
            tts.speak("An error occurred while processing your request.")

        print("="*50 + "\n")

cap.release()
cv2.destroyAllWindows()