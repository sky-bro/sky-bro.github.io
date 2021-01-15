 The **Curses** library forms a wrapper over working with raw terminal codes, and provides highly flexible and efficient API (Application Programming Interface). It provides functions to move the cursor, create windows, produce colors, play with mouse etc. The application programs need not worry about the underlying terminal capabilities

<!--more-->

## commonly used

### initialization

```c++
initscr(); // init screen, first thing to to before you using ncurses
cbreak(); // disable the buffering of typed characters by the TTY driver and get a character-at-a-time input
noecho(); // To suppress the automatic echoing of typed characters
curs_set(0);
nodelay(stdscr, TRUE); // getch() return ERR if the key input is not ready
endwin(); // Before the program is terminated, endwin() must be called to restore the terminal settings.
```

### windows

```c++
keypad(stdscr, TRUE); // capture special keystrokes like Backspace, Delete and the four arrow keys by getch()
// new windows
int h, w;
getmaxyx(stdscr, h, w); // get the size of the screen.
WINDOW * win = newwin(nlines, ncols, y0, x0); // y0 and x0 are the coordinates of the upper left corner of win on the screen
// Windows cannot overlap with each other. Therefore you have two options: only use stdscr and no other windows, or create several non-overlapping windows but do not use stdscr.
wrefresh(win); // refresh() is equivalent to wrefresh(stdscr).
delwin(win); // If a window win is no longer needed, and you're going to create new windows to overlap it, you should call delwin(win) to delete the window (release the memory it is using).
box
derwin
```

### input

```c++
wmove(win, y, x); // move(y, x) is equivalent to the wmove(stdscr, y, x). The actual cursor motion is not shown on the screen untill you do a wrefresh(win).
wgetch(win); // The user input of course comes from the keyboard and not the screen window. But the different windows on the screen might have different delay modes and other properties, therefore affect the behavior of wgetch()
mvwgetch(win); // mvgetch()
```

### output

```c++
waddch(win, y, x, ch); // addch();
mvwaddch();
wechochar(win, ch); // waddch(win, ch); wrefresh(win); 
waddstr(win, str);
wprintw(win, fmtstr, arg1, arg2, ...);
```

## more

add more as I go...

## refs

* [Ncurses Programming Guide by X. Li](http://www.cs.ukzn.ac.za/~hughm/os/notes/ncurses.html) -- simple and quick
* [NCURSES Programming HOWTO](https://tldp.org/HOWTO/NCURSES-Programming-HOWTO/) -- full
* [go tui libs](https://appliedgo.net/tui/)
