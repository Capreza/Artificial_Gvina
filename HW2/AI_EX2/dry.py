



a)
function Minimax(State):
    if G(State) then return U(State)
    Turn <- Turn(State)
    Children <- Succ(State)
    CurMax <- [-inf,-inf, ... , -inf] # vector size = amount of players
    Loop for c in Children
            v <- Minimax(c)
            if CurMax[Turn] < v[Turn] then CurMax <- v
    Return(CurMax)

explanation:
v is the minimax value of the current turn,
it is a vector of utility scores for each player
v[Turn] is the utility score for the current player


b)
function Minimax(State,Agent):
    if G(State) then return U(State,Agent)
    Turn <- Turn(State)
    Children <- Succ(State)
    if Turn = Agent then:
        CurMax <- inf
        Loop for c in Children
                v <- Minimax(c)
                if CurMax[Turn] < v[Turn] then CurMax <- v
        Return(CurMax)
    else:
        CurMin <- inf
        Loop for c in Children
            v <- Minimax(c)
            if CurMin < v then CurMin <- v
        Return(CurMin)

explanation:
It is as if there are 2 groups, the agent and everyone else
and so the code is the same as regular minimax between two agents, where each one
wants to maximize its own utility on the expense of the other. It is a zero sum game
between the two groups (the agent, all other agents).

c)
function Minimax(State,Agent):
    if G(State) then return U(State)
    Turn <- Turn(State)
    Children <- Succ(State)
    CurMax <- [-inf,-inf, ... , -inf] # vector size = amount of players
    Loop for c in Children
            v <- Minimax(c)
            if CurMax[(Turn+1)%n] < v[(Turn+1)%n] then CurMax <- v # n is amount of players
    Return(CurMax)

explanation:
Only difference between this one and seif a, is that we are looking to maximize
the utility of Turn+1 instead of Turn.