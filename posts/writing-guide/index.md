
## Create new post

First you need to create a file for you to write. Here's how you can create a new post:

```shell
hugo new posts/writing-guide.md # will create content/en/posts/writing-guide.md, 'en' is set as my site's default language
hugo new posts/writing-guide.zh.md # content/zh/posts/writing-guide.zh.md
hugo new posts/writing-guide/index.md # content/en/posts/writing-guide/index.md
hugo new posts/writing-guide/index.zh.md # content/en/posts/writing-guide/index.zh.md
```

I prefer to create a directory for each of my writings, so I can add extra stuff like images for that specific page more easily.

## front matter

hugo supports four front matter formats: yaml, toml, json and org. [hugo: Front Matter Formats](https://gohugo.io/content-management/front-matter/#front-matter-formats)

I prefer using yaml, because I think it's easier to understand compared with the other three. I suggest you only use yaml or toml.

```markdown
---
# yaml front matter
---
```

```markdown
+++
# toml front matter
+++
```

## Code

you can write multi-lines of code

`````markdown
````markdown
# markdown code example
keep nesting with different # of ticks (the more to the outside, the more ticks)
```markdown
## some header

* list item1
* list item2
```
````
`````

## Pictures

```markdown
<!-- no caption by default -->
![an image in this directory](./images/scott.jpg)
```

![an image in this directory](./images/scott.jpg)

```markdown
<!-- better way, use figure shortcode -->
<!-- to know more shortcodes: https://gohugo.io/content-management/shortcodes/ -->
{{</* figure src="./images/scott.jpg" caption="an image in this directory (this is Scott Adkins)" alt="an image in this directory" */>}}
```

{{< figure src="./images/scott.jpg" caption="an image in this directory (this is Scott Adkins)" alt="an image in this directory" >}}

## Math Support (with MathJax)

You can get some quick examples from: [typora's doc](https://support.typora.io/Math/)

```markdown
single \\$ for inline code: $x,y,z$
two \\$'s for block level code:

$$z = \sqrt{x^2 + y^2}$$

$$
\begin{align*}
y = y(x,t) &= A e^{i\theta} \\\\
&= A (\cos \theta + i \sin \theta) \\\\
&= A (\cos(kx - \omega t) + i \sin(kx - \omega t)) \\\\
&= A\cos(kx - \omega t) + i A\sin(kx - \omega t)  \\\\
&= A\cos \Big(\frac{2\pi}{\lambda}x - \frac{2\pi v}{\lambda} t \Big) + i A\sin \Big(\frac{2\pi}{\lambda}x - \frac{2\pi v}{\lambda} t \Big)  \\\\
&= A\cos \frac{2\pi}{\lambda} (x - v t) + i A\sin \frac{2\pi}{\lambda} (x - v t)
\end{align*}
$$

```

single \\$ for inline code: $x,y,z$
two \\$'s for block level code:

$$z = \sqrt{x^2 + y^2}$$

$$
\begin{align*}
y = y(x,t) &= A e^{i\theta} \\\\
&= A (\cos \theta + i \sin \theta) \\\\
&= A (\cos(kx - \omega t) + i \sin(kx - \omega t)) \\\\
&= A\cos(kx - \omega t) + i A\sin(kx - \omega t)  \\\\
&= A\cos \Big(\frac{2\pi}{\lambda}x - \frac{2\pi v}{\lambda} t \Big) + i A\sin \Big(\frac{2\pi}{\lambda}x - \frac{2\pi v}{\lambda} t \Big)  \\\\
&= A\cos \frac{2\pi}{\lambda} (x - v t) + i A\sin \frac{2\pi}{\lambda} (x - v t)
\end{align*}
$$

## Flow charts

TODO

## More

To be added...
