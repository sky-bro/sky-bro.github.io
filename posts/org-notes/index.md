
## Basic Editing {#basic-editing}


### Comments {#comments}

_**C-c ;**_
: toggle comment of an entry


### Font types {#font-types}

```org
+ /italic/
+ *bold*
+ _underlined_
+ =verbatim=
+ ~code~
+ +strike-through+
```

will be rendered as:

-   _italic_
-   **bold**
-   <span class="underline">underlined</span>
-   `verbatim`
-   `code`
-   ~~strike through~~


## Code {#code}

Offers two types of source code:

1.  code block
2.  inline code

org-entities-help function helps you insert some code.


### inline {#inline}

```org
src_c++[:exports code]{ typedef long long ll; }
src_shell[:exports code]{ echo -e "test" }
```

` typedef long long ll; `
` echo -e "test" `


### code block {#code-block}

source code blocks are one of many Org block types.

```org
#+BEGIN_SRC cpp
  #include <iostream>
  using namespace std;
  int main() {
    cout << "123\n";
    return 0;
  }
#+END_SRC
```

```cpp
#include <iostream>
using namespace std;
int main() {
  cout << "123\n";
  return 0;
}
```


## List {#list}

M-RET
: new item at current level

M-S-RET
: new item with a checkbox

M-UP/DOWN
: move item up/down, including subitems

M-S-UP/DOWN
: move item up/down

M-LEFT/RIGHT
: decrease/increase indentation of item

M-S-LEFT/RIGHT
: decrease/increase indentation of item, including subitems

C-c C-c
: toggle checkbox

C-c -
: Cycle through itemize/enumerate bullets


## Table {#table}

-   _**|Name|Age C-c RET**_ create table with headers

    | NAME | Age |
    |------|-----|
    | sky  | 22  |
    | k4i  | 23  |
-   _**RET**_ go to next row
-   _**S-UP/DOWN/LEFT/RIGHT**_ swap between cells
-   _**M-UP/DOWN/LEFT/RIGHT**_ swap between rows/columns
-   _**M-S-UP/DOWN/LEFT/RIGHT**_ insert/delete row/column
-   _**C-c -**_ insert horizontal line below
-   _**C-c RET**_ insert horizontal line below, move to next row
-   _**C-c ^**_ sort column


## Footnote {#footnote}

for more information on footnote, please refer to the official org site[^fn:1].


### footnote types: {#footnote-types}

named footnote
: fn:NAME

anonymous, inline footnote
: fn:: inline definition, fn:NAME: inline definition


### example {#example}

```org
The Org homepage[fn:1] now looks a lot better than it used to.
...
[fn:1] The link is: https://orgmode.org
```


## hyperlinks {#hyperlinks}

-   formats
    -   `[[link][description]]`
    -   `[[link]]`
    -   [k4i's home!](https://k4i.top/)
-   link types
    -   internal links
    -   external links
-   shortcuts
    -   **_**C-c C-l**_:** insert/delete link
    -   **_**C-c C-o**_:** open link


## todos <code>[1/2]</code> {#todos}


### <span class="org-todo done DONE">DONE</span> subtask 01 {#subtask-01}

_**M-S-RET**_
: new todo item

_**C-c C-t**_
: cycle through todo states


### BUG subtask 02 <code>[1/2]</code> {#bug-subtask-02}

-   [-] item 01
    -   [ ] item 01.01
    -   [X] item 01.02
-   [X] item 02


## Images {#images}

_**C-c C-x C-v**_
: toggle images (org-toggle-inline-images)


## Exports {#exports}


### latex {#latex}

latex config

```shell
tlmgr update elegantpaper
tlmgr install elegantpaper # [[https://github.com/ElegantLaTeX/ElegantPaper][elegantpaper]]
tlmgr uninstall elegantpaper
pip install pygments # dependency of [[https://github.com/gpoore/minted][minted]]
```

add this in your front matter

```org
#+LATEX_COMPILER: xelatex
#+LATEX_CLASS: elegantpaper
#+OPTIONS: prop:t
```

[^fn:1]: [org mode official site](https://orgmode.org/)
