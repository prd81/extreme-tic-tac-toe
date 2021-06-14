import sys
import random
import signal
import copy
import time

class AI_Bot:

    def __init__(self):
        signal.signal(signal.SIGINT, self.shutdown)     # Shutdown on Ctrl+C
        self.maxdepth = 5

        self.current_depth = 1

        self.current_count = 0

        self.came = 0

        self.max_time = 15

        self.timed_out = False;

        self.inf = 1000000000000

        self.threshold = [3, 6.83, 4.83, 0]

        self.win_pos = [
                ( 0, 1, 2, 3),
                ( 4, 5, 6, 7),
                ( 8, 9,10,11),
                (12,13,14,15),
                ( 0, 4, 8,12),
                ( 1, 5, 9,13),
                ( 2, 6,10,14),
                ( 3, 7,11,15),
                ( 1, 4, 6, 9),
                ( 2, 5, 7,10),
                ( 5, 8,10,13),
                ( 6, 9,11,14),
                ]

        self.twos = []

        self.threes = []

        for each in self.win_pos:
            self.twos.append((each[0],each[1],each[2],each[3]))
            self.twos.append((each[0],each[2],each[1],each[3]))
            self.twos.append((each[0],each[3],each[1],each[2]))
            self.twos.append((each[1],each[2],each[0],each[3]))
            self.twos.append((each[1],each[3],each[0],each[2]))
            self.twos.append((each[2],each[3],each[0],each[1]))
            self.threes.append((each[0],each[1],each[2],each[3]))
            self.threes.append((each[1],each[2],each[3],each[0]))
            self.threes.append((each[2],each[3],each[0],each[1]))
            self.threes.append((each[3],each[0],each[1],each[2]))


        self.corners = [ 0, 3,12,15]
        self.centers = [ 5, 6, 9,10]
        self.rest    = [ 1, 2, 4, 7, 8,11,13,14]
        self.block_value = [[2, 3, 3, 2], [3, 4, 4, 3], [3, 4, 4, 3], [2, 3, 3, 2]]

        self.score = {'x':1, 'o':1, '-':1, 'd':1}

        self.cell_score = [[], [], [], []]

        self.flag = " "

        self.opp_flag = " "

        self.heur_dict = {}

    def shutdown(self, signum, frame):
        print
        sys.exit(0)

    def blocks_allowed(self, old_move, block_stat):
        blocks = []
        if old_move == (-1, -1):
            final_blocks_allowed = [5]

        blocks = [4*((old_move[0]%4)) + ((old_move[1]%4))]
        if block_stat[blocks[0]/4][blocks[0]%4] == '-':
            return blocks
        final_blocks_allowed = []

        for block in range(16):
            if block_stat[block/4][block%4] == '-':
                final_blocks_allowed.append(block)

        return final_blocks_allowed

    def cells_allowed(self, temp_board, blocks_allowed, block_stat):
        cells = []

        for block in blocks_allowed:

            start_row = (block / 4) * 4
            start_col = ((block) % 4) * 4

            for i in range(start_row, start_row + 4):
                for j in range(start_col, start_col + 4):
                    if temp_board[i][j] == '-':
                        cells.append((i,j))

        if not cells:
            for i in range(16):
                if block_stat[i/4][i%4] != '-':
                    continue
                start_row = (i / 4) * 4
                start_col = ((i) % 4) * 4
                for j in range(start_row, start_row + 4):
                    for k in range(start_col, start_col + 4):
                        if temp_board[j][k] == '-':
                            cells.append((j,k))
        return cells

    def heuristic(self, node, temp_block):

        utility = self.getBoardScore(node, temp_block)

        return utility

    def getBlockStatus(self, block):
    	dLis = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    	for k in (self.flag, self.opp_flag):
    		row = any(all(block[i][j]==k for i in range(4)) for j in range(4))
    		col = any(all(block[i][j]==k for j in range(4)) for i in range(4))
    		dim = any(all(block[i+x+1][j+y+1]==k for (x,y) in dLis) for i in range(2) for j in range(2))

    		if row or col or dim:
    			return k

    	if [block[i].count('-') for i in range(4)].count(0) == 4:
    		return 'd'

    	return '-'

    def getBlockScore(self, block):
        block_tuple = tuple([tuple(block[i]) for i in range(4)])
        def util():
            best_score  = -100000
            moves = []
            for i in range(4):
                for j in range(4):
                    if block[i][j] == '-':
                        moves.append((i, j))
            for move in moves:
                score = 1 + self.getScore(move[0], move[1], block)
                best_score = max(best_score, score)
            return best_score
        val = {self.flag:100, self.opp_flag:0, 'd':0}
        if block_tuple not in self.heur_dict:
            self.heur_dict[block_tuple] = val.get(self.getBlockStatus(block), util())
        return self.heur_dict[block_tuple]

    def getScore(self, a, b, block):
        score = 0
        token_val = {self.flag : 1, self.opp_flag : 0, '-' : 1, 'd' : 0}
        row = block[a]
        if not self.opp_flag in row:
            temp_val = []

        col = [block[i][b] for i in range(4)]
        if not self.opp_flag in col:
        	score += 2<<col.count(self.flag)

        dLis = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        cLis = [[(i+x+1, j+y+1) for (x,y) in dLis] for i in range(2) for j in range(2)]

        tLis = list(filter(lambda l: (a,b) in l,cLis))

        for dim in tLis:
        	vset = [block[i][j] for (i,j) in dim]
        	if not self.opp_flag in vset:
        		score += 2<<vset.count(self.flag)

        row = block[a]
        if not self.flag in row:
        	score += 2<<row.count(self.opp_flag)
        col = [block[i][b] for i in range(4)]
        if not self.flag in col:
        	score += 2<<col.count(self.opp_flag)

        dLis = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        cLis = [[(i+x+1, j+y+1) for (x,y) in dLis] for i in range(2) for j in range(2)]

        tLis = list(filter(lambda l: (a,b) in l,cLis))

        for dim in tLis:
        	vset = [block[i][j] for (i,j) in dim]
        	if not self.flag in vset:
        		score += 2<<vset.count(self.opp_flag)

        return score

    def lineScore(self, line, curr_block, curr_opp_block, current_block_status):
        if 'd' in [current_block_status[x][y] for (x,y) in line]:
            return 0
        pos = [curr_block[x][y] for (x,y) in line]
        neg = [curr_opp_block[x][y] for (x,y) in line]
        positive = 1
        negative = 1
        for i in pos:
            positive *= i
        for i in neg:
            negative *= i
        return positive - negative

    def getTerminalStatus(self, node, temp_block):
        terminalStatus = self.getBlockStatus(temp_block)
        if terminalStatus == '-':
            return (False, 0)
        if terminalStatus == self.flag:
            return (True, 1000000000000)
        if terminalStatus == self.opp_flag:
            return (True, -1000000000000)
        score = 0
        score_dict = {self.flag : 1, self.opp_flag : -1}
        for cell in self.centers:
            score += score_dict.get(temp_block[cell/4][cell%4], 0)*3
        for cell in self.corners:
            score += score_dict.get(temp_block[cell/4][cell%4], 0)*6
        for cell in self.rest:
            score += score_dict.get(temp_block[cell/4][cell%4], 0)*4
        return (True, score)

    def getBoardScore(self, node, temp_block):
        terminalStatus, terminalScore = self.getTerminalStatus(node, temp_block)
        if terminalStatus:
            return terminalScore
        opp_node = copy.deepcopy(node)
        for i in range(16):
            for j in range(16):
                if node[i][j] == self.flag:
                    opp_node[i][j] = self.opp_flag
                if node[i][j] == self.opp_flag:
                    opp_node[i][j] = self.flag

        node_score = [[0]*4 for i in range(4)]
        opp_node_score = [[0]*4 for i in range(4)]

        for i in range(4):
            for j in range(4):
                temp = [[node[x][y] for y in range(4*j, 4*j + 4)] for x in range(4*i, 4*i + 4)]
                node_score[i][j] = self.getBlockScore(temp)
                temp = [[opp_node[x][y] for y in range(4*j, 4*j + 4)] for x in range(4*i, 4*i + 4)]
                opp_node_score[i][j] = self.getBlockScore(temp)

        boardScore = []
        for i in range(4):
            line = [(i, j) for j in range(4)]
            boardScore.append(self.lineScore(line, node_score, opp_node_score, temp_block))
            line = [(j, i) for j in range(4)]
            boardScore.append(self.lineScore(line, node_score, opp_node_score, temp_block))

        dLis = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        cLis = [[(i+x+1, j+y+1) for (x,y) in dLis] for i in range(2) for j in range(2)]

        for line in cLis:
            boardScore.append(self.lineScore(line, node_score, opp_node_score, temp_block))

        if 1000000000000 in boardScore:
            return 1000000000000
        if -1000000000000 in boardScore:
            return -1000000000000
        return sum(boardScore)


    def genChild(self, node, temp_block, mov, current_flag):
        temp_node = copy.deepcopy(node)
        temp_node[mov[0]][mov[1]] = current_flag
        current_temp_block = copy.deepcopy(temp_block)

        block_num = (mov[0] / 4) * 4 + (mov[1] / 4)

        temp_stat = []
        start_row = (block_num / 4) * 4
        start_col = ((block_num) % 4) * 4
        for j in range(start_row, start_row + 4):
            for k in range(start_col, start_col + 4):
                temp_stat.append(temp_node[j][k])

        for each in self.win_pos:
            if temp_stat[each[0]] == self.flag and temp_stat[each[1]] == self.flag and temp_stat[each[2]] == self.flag and temp_stat[each[3]] == self.flag:
                current_temp_block[block_num/4][block_num%4] = self.flag
                break
            if temp_stat[each[0]] == self.opp_flag and temp_stat[each[1]] == self.opp_flag and temp_stat[each[2]] == self.opp_flag and temp_stat[each[3]] == self.opp_flag:
                current_temp_block[block_num/4][block_num%4] = self.opp_flag
                break

        return (temp_node, current_temp_block)


    def alphabeta(self, node, depth, alpha, beta, maximizingPlayer, old_move, temp_block):
        if (depth == 0 and depth != self.current_depth) or self.timed_out == True:
                return self.heuristic(copy.deepcopy(node), copy.deepcopy(temp_block)), ()

        if time.time() - self.came > self.max_time:
            self.timed_out = True
            return self.heuristic(copy.deepcopy(node), copy.deepcopy(temp_block)), ()

        blocks = self.blocks_allowed(old_move, temp_block)

        cells_allowed = self.cells_allowed(node, blocks, temp_block)

        if not cells_allowed:
            return self.heuristic(copy.deepcopy(node), copy.deepcopy(temp_block)), ()

        ret_mov = " "
        random.shuffle(cells_allowed)
        if maximizingPlayer:
            v = -self.inf
            ret_mov = cells_allowed[0]
            for mov in cells_allowed:
                tmp = self.genChild(node, temp_block, mov, self.flag)
                child = tmp[0]
                current_temp_block = tmp[1]

                temp, temp_move = self.alphabeta(copy.deepcopy(child), depth - 1, copy.deepcopy(alpha), copy.deepcopy(beta), False, copy.deepcopy(mov), copy.deepcopy(current_temp_block))

                if v < temp:
                    v = temp
                    ret_mov = mov
                alpha = max(alpha, v)

                if time.time() - self.came > self.max_time:
                    self.timed_out = True
                    self.print_data("case of timeout value", v)
                    self.print_data("case of timeout move", ret_mov)
                    return v, ret_mov

                if beta <= alpha:
                    break
            return v, ret_mov

        else:
            v = self.inf
            ret_mov = cells_allowed[0]
            for mov in cells_allowed:
                tmp = self.genChild(node, temp_block, mov, self.opp_flag)
                child = tmp[0]
                current_temp_block = tmp[1]

                temp, temp_move = self.alphabeta(copy.deepcopy(child), depth - 1, copy.deepcopy(alpha), copy.deepcopy(beta), True, copy.deepcopy(mov), copy.deepcopy(current_temp_block))

                if v > temp:
                    v = temp
                    ret_mov = mov
                beta = min(beta, v)

                if time.time() - self.came > self.max_time:
                    self.timed_out = True
                    self.print_data("case of timeout value", v)
                    self.print_data("case of timeout move", ret_mov)
                    return v, ret_mov

                if beta <= alpha:
                    break
            return v, ret_mov

    def print_data(self, title, data):
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print(title, data)
        print("XXXXXXXXXXXXXXXXXXXXXXXX")

    def move(self, current_board, old_move, flag):
        if old_move == (-1, -1):
            return (5, 5)
        temp_block = current_board.block_status
        temp_board = current_board.board_status
        ret2 = " "
        ret3 = " "
        ret4 = " "
        self.timed_out = False
        self.flag = flag
        if self.opp_flag == " ":
            if self.flag == 'x':
                self.opp_flag = 'o'
            else:
                self.opp_flag = 'x'
        self.score[self.flag] = 2
        self.score[self.opp_flag] = 0
        self.current_depth = 1
        self.came = time.time()
        ret_score, ret = self.alphabeta(copy.deepcopy(temp_board), self.current_depth,  -self.inf, self.inf, True, copy.deepcopy(old_move), copy.deepcopy(temp_block))
        self.print_data("for depth ", self.current_depth)
        self.print_data("returned move is ", ret)
        self.print_data("returned move score is ", ret_score)
        self.print_data("timed_out = ", self.timed_out)
        while not self.timed_out and self.current_depth < self.maxdepth:
            self.current_depth += 1
            ret_score1, ret1 = self.alphabeta(copy.deepcopy(temp_board), self.current_depth,  -self.inf, self.inf, True, copy.deepcopy(old_move), copy.deepcopy(temp_block))
            self.print_data("for depth ", self.current_depth)
            self.print_data("returned move is ", ret1)
            self.print_data("returned move score is ", ret_score1)
            self.print_data("timed_out = ", self.timed_out)
            if ret_score1 > ret_score:
                ret_score = ret_score1
                ret = ret1
            if self.timed_out:
                break
        self.print_data("finally returned move is ", ret)
        self.print_data("finally returned move score is ", ret_score)
        return ret
