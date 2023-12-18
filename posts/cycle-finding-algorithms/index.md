
## Floyd's tortoise and hare {#floyd-s-tortoise-and-hare}

{{< figure src="/images/posts/cycle-finding/tortoise-and-hare.svg" caption="<span class=\"figure-number\">Figure 1: </span>tortoise and hare" >}}

hare and tortoise both starts from point **Start**, at speed of \\(2\\) and \\(1\\) respectively. They'll meet at some point in the cycle.

when they meet, hare traveled \\(S\_2 - S\_{1}\\) more points than tortoise. We have: (\\(\lambda\\) is the cycle length)

\\[
S\_{2} - S\_{1} = 0  = S\_{1} = x + m \pmod{\lambda}
\\]

after they meet, one starts over from Start point, the other starts from Meet point, at speed of \\(1\\), they'll meet again at **Cycle Start** point. because:

\\[
   x = x + m + x \pmod{\lambda}
   \\]
