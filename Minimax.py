import math
from typing import Dict, List, Tuple, Optional, Literal

# Type for piece colors
PieceColor = Literal["white", "black"]

class MinimaxChessAI:
    # Evaluation constants
    CHECKMATE_BONUS = 1000000000
    ROOK_THREATENED_PENALTY = -200
    ROOK_DEFENDED_BONUS = 50
    ROOK_ATTACKS_KING_BONUS = 50
    ROOK_MOBILITY_WEIGHT = 1.0
    KING_PROXIMITY_WEIGHT = 3.0
    EDGE_CONFINEMENT_WEIGHT = 15.0
    ROOK_CUTOFF_BONUS = 100

    def __init__(self, depth: int = 3):
        """Initialize the AI with search depth."""
        if depth < 1:
            raise ValueError("Depth must be at least 1")
        self.depth = depth

    def get_best_move(self, piece_positions: Dict[str, str]) -> Tuple[str, str]:
        """
        Find the best move for black pieces (king or rook) 
        using Minimax with alpha-beta pruning.
        """
        print(f"Calculating best move with depth {self.depth}...")
        best_move, best_eval = self._minimax(piece_positions, self.depth, -math.inf, math.inf, True)

        if best_move:
            print(f"Best move found: {best_move[0]} -> {best_move[1]} with evaluation: {best_eval:.2f}")
            return best_move
        else:
            # Fallback if no move found (shouldn't happen if legal moves exist)
            print("Minimax didn't find preferred move, looking for any legal move...")
            moves = self._generate_legal_moves(piece_positions, "black")
            if moves:
                print("Returning first legal move as fallback.")
                origin = next(iter(moves))
                target = moves[origin][0]
                return (origin, target)
            else:
                print("Warning! No legal moves found for black.")
                king_pos = piece_positions.get("black_king", "a1")
                return (king_pos, king_pos)

    def _minimax(self, state: Dict[str, str], depth: int, alpha: float, beta: float, 
                is_maximizing_player: bool) -> Tuple[Optional[Tuple[str, str]], float]:
        """
        Minimax algorithm with alpha-beta pruning.
        Returns best move (if applicable) and state evaluation.
        """
        current_color: PieceColor = "black" if is_maximizing_player else "white"
        legal_moves_dict = self._generate_legal_moves(state, current_color)
        is_terminal_state = not bool(legal_moves_dict)

        if is_terminal_state or depth == 0:
            return None, self._evaluate(state, current_color, is_terminal_state)

        # Flatten move dictionary to list of (origin, destination) tuples
        all_possible_moves: List[Tuple[str, str]] = []
        for origin, destinations in legal_moves_dict.items():
            for dest in destinations:
                all_possible_moves.append((origin, dest))

        if not all_possible_moves:
            return None, self._evaluate(state, current_color, True)

        best_move_found = all_possible_moves[0] if all_possible_moves else None

        if is_maximizing_player:  # Black's turn (maximizer)
            best_value = -math.inf
            for move in all_possible_moves:
                new_state = self._apply_move(state, move, current_color)
                _, value = self._minimax(new_state, depth - 1, alpha, beta, False)

                if value > best_value:
                    best_value = value
                    best_move_found = move

                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # Beta cutoff

            return best_move_found, best_value

        else:  # White's turn (minimizer)
            best_value = math.inf
            for move in all_possible_moves:
                new_state = self._apply_move(state, move, current_color)
                _, value = self._minimax(new_state, depth - 1, alpha, beta, True)

                if value < best_value:
                    best_value = value
                    best_move_found = move

                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Alpha cutoff

            return best_move_found, best_value

    def _generate_legal_moves(self, state: Dict[str, str], color: PieceColor) -> Dict[str, List[str]]:
        """
        Generate all legal moves for given color.
        Key: origin position, Value: list of destination positions.
        Filters moves that would leave king in check.
        """
        moves: Dict[str, List[str]] = {}
        opponent_color: PieceColor = "white" if color == "black" else "black"

        # Generate pseudo-legal moves (without considering own king checks)
        pseudo_legal_moves: Dict[str, List[str]] = {}
        if color == "black":
            king_pos = state.get("black_king")
            rook_pos = state.get("black_rook")
            if king_pos:
                king_moves = self._get_pseudo_legal_king_moves(king_pos, state, color)
                if king_moves: pseudo_legal_moves[king_pos] = king_moves
            if rook_pos:
                rook_moves = self._get_pseudo_legal_rook_moves(rook_pos, state, color)
                if rook_moves: pseudo_legal_moves[rook_pos] = rook_moves
        else:  # color == "white"
            king_pos = state.get("white_king")
            if king_pos:
                king_moves = self._get_pseudo_legal_king_moves(king_pos, state, color)
                if king_moves: pseudo_legal_moves[king_pos] = king_moves

        # Filter moves that put own king in check
        for origin, destinations in pseudo_legal_moves.items():
            legal_destinations = []
            current_king_key = f"{color}_king"

            for dest in destinations:
                temp_state = self._apply_move(state, (origin, dest), color)
                king_after_move_pos = temp_state.get(current_king_key)

                if king_after_move_pos and not self._is_square_attacked(king_after_move_pos, temp_state, opponent_color):
                    legal_destinations.append(dest)

            if legal_destinations:
                moves[origin] = legal_destinations

        return moves

    def _get_pseudo_legal_king_moves(self, king_pos: str, state: Dict[str, str], color: PieceColor) -> List[str]:
        """Calculate pseudo-legal moves for a king (without checking own king checks)."""
        col_char, row_int = king_pos[0], int(king_pos[1])
        moves = []
        opponent_color: PieceColor = "white" if color == "black" else "black"

        friendly_pieces = set(pos for key, pos in state.items() if key.startswith(color))
        opponent_king_pos = state.get(f"{opponent_color}_king")

        # Possible king moves (8 directions)
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dc, dr in deltas:
            new_col_ord = ord(col_char) + dc
            new_row = row_int + dr

            if ord('a') <= new_col_ord <= ord('h') and 1 <= new_row <= 8:
                new_pos = f"{chr(new_col_ord)}{new_row}"

                if new_pos in friendly_pieces:
                    continue

                if opponent_king_pos and self._are_squares_adjacent(new_pos, opponent_king_pos):
                    continue

                moves.append(new_pos)

        return moves

    def _get_pseudo_legal_rook_moves(self, rook_pos: str, state: Dict[str, str], color: PieceColor) -> List[str]:
        """Calculate pseudo-legal moves for a rook."""
        col_char, row_int = rook_pos[0], int(rook_pos[1])
        moves = []
        opponent_color: PieceColor = "white" if color == "black" else "black"

        friendly_pieces = set(pos for key, pos in state.items() if key.startswith(color))
        opponent_pieces = set(pos for key, pos in state.items() if key.startswith(opponent_color))

        # Rook movement directions: horizontal and vertical
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left

        for dc, dr in directions:
            for step in range(1, 8):  # Move up to 7 squares
                new_col_ord = ord(col_char) + dc * step
                new_row = row_int + dr * step

                if not (ord('a') <= new_col_ord <= ord('h') and 1 <= new_row <= 8):
                    break  # Off board

                new_pos = f"{chr(new_col_ord)}{new_row}"

                if new_pos in friendly_pieces:
                    break  # Blocked by friendly piece

                moves.append(new_pos)

                if new_pos in opponent_pieces:
                    break  # Capture enemy piece

        return moves

    def _are_squares_adjacent(self, pos1: Optional[str], pos2: Optional[str]) -> bool:
        """Check if two squares are adjacent (including diagonals). Handles None."""
        if not pos1 or not pos2:
            return False
        col1, row1 = ord(pos1[0]), int(pos1[1])
        col2, row2 = ord(pos2[0]), int(pos2[1])
        return max(abs(col1 - col2), abs(row1 - row2)) <= 1

    def _apply_move(self, state: Dict[str, str], move: Tuple[str, str], color: PieceColor) -> Dict[str, str]:
        """
        Create new state by applying a move for given color.
        Updates moved piece position and removes captured pieces.
        """
        new_state = state.copy()
        origin, dest = move

        moved_piece_key = None
        for key, pos in state.items():
            if key.startswith(color) and pos == origin:
                moved_piece_key = key
                break

        if moved_piece_key:
            new_state[moved_piece_key] = dest

            opponent_color = "white" if color == "black" else "black"
            captured_piece_key = None
            for key, pos in state.items():
                if key.startswith(opponent_color) and pos == dest:
                    captured_piece_key = key
                    break
            if captured_piece_key:
                del new_state[captured_piece_key]
        else:
            print(f"Critical Error: No {color} piece found at origin {origin} for state {state} and move {move}")
            pass

        return new_state

    def _is_square_attacked(self, target_pos: str, state: Dict[str, str], attacker_color: PieceColor) -> bool:
        """Check if target_pos is attacked by any attacker_color piece."""
        opponent_king_pos = state.get(f"{attacker_color}_king")
        if opponent_king_pos and self._are_squares_adjacent(target_pos, opponent_king_pos):
            return True

        opponent_rook_pos = state.get(f"{attacker_color}_rook")
        if opponent_rook_pos:
            defender_color: PieceColor = "white" if attacker_color == "black" else "black"
            friendly_king_pos = state.get(f"{defender_color}_king")
            if self._is_square_attacked_by_rook(target_pos, opponent_rook_pos, friendly_king_pos, state):
                return True

        return False

    def _evaluate(self, state: Dict[str, str], current_player_color: PieceColor, is_terminal: bool) -> float:
        """
        Evaluate how good a state is FROM BLACK'S PERSPECTIVE (maximizer).
        High score favors black, low score favors white.
        """
        score = 0.0

        white_king_pos = state.get("white_king")
        black_king_pos = state.get("black_king")
        black_rook_pos = state.get("black_rook")

        # --- Terminal State Checks ---
        if is_terminal:
            king_to_check = state.get(f"{current_player_color}_king")
            opponent_color: PieceColor = "white" if current_player_color == "black" else "black"

            if king_to_check and self._is_square_attacked(king_to_check, state, opponent_color):
                if current_player_color == "white":
                    return self.CHECKMATE_BONUS
                else:
                    return -self.CHECKMATE_BONUS
            else:
                return 0.0  # Stalemate

        # --- Heuristic Evaluation ---
        if not white_king_pos: return self.CHECKMATE_BONUS
        if not black_king_pos or not black_rook_pos: return -self.CHECKMATE_BONUS

        # 1. Black Rook Safety
        is_rook_threatened = self._are_squares_adjacent(white_king_pos, black_rook_pos)
        if is_rook_threatened:
            score += self.ROOK_THREATENED_PENALTY
            is_rook_defended = self._are_squares_adjacent(black_king_pos, black_rook_pos)
            if is_rook_defended:
                score += self.ROOK_DEFENDED_BONUS

        # 2. Black Rook Attacking White King
        is_king_attacked = self._is_square_attacked_by_rook(white_king_pos, black_rook_pos, black_king_pos, state)
        if is_king_attacked:
            score += self.ROOK_ATTACKS_KING_BONUS

        # 3. King Proximity
        king_distance = self._manhattan_distance(white_king_pos, black_king_pos)
        score += (14.0 - king_distance) * self.KING_PROXIMITY_WEIGHT

        # 4. White King Edge Confinement
        w_king_col, w_king_row = ord(white_king_pos[0]), int(white_king_pos[1])
        dist_to_a_edge = w_king_col - ord('a')
        dist_to_h_edge = ord('h') - w_king_col
        dist_to_1_edge = w_king_row - 1
        dist_to_8_edge = 8 - w_king_row
        
        # Chebyshev distance
        distance_from_edge = min(dist_to_a_edge, dist_to_h_edge, dist_to_1_edge, dist_to_8_edge)
        score += (3.5 - distance_from_edge) * self.EDGE_CONFINEMENT_WEIGHT

        # 5. Rook Cutting Off White King
        col_w_king, row_w_king = ord(white_king_pos[0]), int(white_king_pos[1])
        col_b_rook, row_b_rook = ord(black_rook_pos[0]), int(black_rook_pos[1])
        col_b_king, row_b_king = ord(black_king_pos[0]), int(black_king_pos[1])

        if abs(row_b_rook - row_w_king) == 1:
            if col_b_king == col_w_king and abs(row_b_king - row_w_king) > 1:
                score += self.ROOK_CUTOFF_BONUS / 2
            else:
                score += self.ROOK_CUTOFF_BONUS

        elif abs(col_b_rook - col_w_king) == 1:
            if row_b_king == row_w_king and abs(col_b_king - col_w_king) > 1:
                score += self.ROOK_CUTOFF_BONUS / 2
            else:
                score += self.ROOK_CUTOFF_BONUS

        # 6. Penalty for Black King-Rook Adjacency
        if self._are_squares_adjacent(black_king_pos, black_rook_pos):
            score -= 10

        return score

    def _is_square_attacked_by_rook(self, target_pos: str, rook_pos: str, 
                                   friendly_king_pos: Optional[str], state: Dict[str, str]) -> bool:
        """
        Check if rook at rook_pos attacks target_pos.
        Considers friendly king blocking and enemy king position.
        """
        if not target_pos or not rook_pos:
            return False

        target_col, target_row = ord(target_pos[0]), int(target_pos[1])
        rook_col, rook_row = ord(rook_pos[0]), int(rook_pos[1])

        if target_row != rook_row and target_col != rook_col:
            return False

        rook_color = None
        if state.get("black_rook") == rook_pos: rook_color = "black"
        elif state.get("white_rook") == rook_pos: rook_color = "white"

        if not rook_color: return False

        opponent_color = "white" if rook_color == "black" else "black"
        opponent_king_pos = state.get(f"{opponent_color}_king")

        if target_row == rook_row:  # Same row
            step = 1 if target_col > rook_col else -1
            for c in range(rook_col + step, target_col, step):
                current_check_pos = f"{chr(c)}{target_row}"
                if current_check_pos == friendly_king_pos or current_check_pos == opponent_king_pos:
                    return False
            return True
        else:  # Same column
            step = 1 if target_row > rook_row else -1
            for r in range(rook_row + step, target_row, step):
                current_check_pos = f"{chr(target_col)}{r}"
                if current_check_pos == friendly_king_pos or current_check_pos == opponent_king_pos:
                    return False
            return True

    def _manhattan_distance(self, pos1: Optional[str], pos2: Optional[str]) -> int:
        """Calculate Manhattan distance between two positions. Handles None."""
        if not pos1 or not pos2:
            return 14  # Maximum distance if piece missing
        col1, row1 = ord(pos1[0]), int(pos1[1])
        col2, row2 = ord(pos2[0]), int(pos2[1])
        return abs(col1 - col2) + abs(row1 - row2)