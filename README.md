# Happy-Medium

The paper explores the polarization by media through the means of experimental design and agent-based modeling. 

Code is written in collaboration with Josh Broomberg.

## Model design:
Individuals have an opinion extremity (ideological position), quality and confidence, all ranging [-10; 10].These are initially set based on a normal distribution between a defined min and max. Individuals are alsoconnected to a number of other nodes (their ‘group’), initially randomly assigned with connection strengths1. Connected individuals which are within a 2 unit distance of an individual’s extremity are considered to bethe “in-group” with all others in the “out-group”. At each step, Citizens update their opinion values based oninteractions within their group of connections and update the connections themselves.

● Confidence is modified depending on the ratio of in-group to out-group individuals. More in-group increases higher confidence as per evidence above.

● Quality moves towards the group’s average, based on the analysis in the causal framework on howdebates improve quality.

● Extremity or ideology is a function of confidence and quality. Low confidence people move towardthe group average, with higher quality individuals moving more slowly (as they assess argumentmore carefully). High confidence individuals with low quality opinions move away from theirout-group - representing low quality, polarising debates. High quality, high confidence individualsdo not move.

● At each turn, an Individual adds a friend of a friend randomly. They also reduce their connectionstrength to other nodes based on the ideological gap, removing the connection if the strength falls tozero.

Media have a position (akin to extremity), set according to a normal or manually defined distribution. Their“subscribers” are individuals with extremities within 2 units of their positions. They exert some “influence”on nodes within 4 units of their position - moving these individuals slightly closer to their position (by onetwentieth of the distance). Each turn, they will optimise their position by moving within a range of 6 units tomaximise their subscriber count and minimize competition (caused by being close to other Media nodes’positions).


## Visualizations: 
Polarization with neutral biases leads to modularity without clear opinion separation.
![](ezgif.com-gif-maker.gif)
