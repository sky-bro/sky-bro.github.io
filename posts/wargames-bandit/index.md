## Basic

1. password for user "bandit0" is "bandit0"
2. ssh into the next host, e.g: `ssh -p2220 bandit0@bandit.labs.overthewire.org`
3. find password for next level
4. goto step 2

## Tricks

* hit `<TAB>` for command completion
* `ls -a` to show hidden files (names started with `.`)
* find command
* man command (`man file`)

## Answers

* level0 (password found: bandit0)
* level0->level1: `cat ./readme` (password found: boJ9jbbUNNfktd78OOpsqOltutMc3MY1)
* level1->level2: `cat ./-` (password found: CV1DtqXWVFXTvM2F0k09SHz0YwRINYA9)
* level2->level3: `cat ./spaces\ in\ this\ filename` (password found: UmHadQclWmgdLOKQ3YNgjWxGoRMb5luK)
* level3->level4: `ls -a ./inhere; cat ./inhere/.hidden` (password found: pIwrPrtPN36QITSp3EQaw936yaFoFgAB)
* level4->level5: `file ./inhere/*; cat ./inhere/-file07` (password found: koReBOKuIDDepwhWk7jZC0RTdopnAYKh)
* level5->level6: `find ./ -size 1033c \! -executable -readable; cat ./inhere/mabehere07/.file2` (password found: DXjZPULLxYr17uwoI01bNLQbtFemEgo7)
* level6->level7: `find / -size 33c -user bandit7 -group bandit6; cat /var/lib/dpkg/info/bandit7.password` (password found: HKBPTKQnIay4Fw76bEy8PVxKEDQRKTzs)
* level7->level8: `grep millionth` (password found: cvX2JJa4CFALtqS87jk27qwqGhBM9plV)
* level8->level9: `sort data.txt | uniq -u` (password found: UsvVyFSfZZWbi6wgC7dAFyFuR6jQQUhR)
* level9->level10: `strings data.txt | grep "==="` (password found: truKLdjsbJ5g7yyJ2X2R0o3a5HQJFuLk)
* level10->level11: `base64 -d data.txt` (password found: IFukwKGsFW8MOq3IRFqrxE1hxTNEbUPR)
* level11->level12: `cat data.txt | tr "a-zA-Z" "n-za-mN-ZA-M"` (password found: 5Te8Y4drgCRfCx8ugdwuEX8KFC6k2EUu)
* level12->level13: `xxd -r data.txt > data; file data; ...` (password found: 8ZjyCRiBWFYkneahHwxCv3wb2a1ORpYL)
* level13->level14: `ssh -i sshkey.private bandit14@localhost` (password found: 4wcYUJFw0k0XLShlDzztnTBHiqxU3b3e)
* level14->level15: `nc localhost 30000` (password found: BfMYroe26WYalil77FoDi9qh59eK5xNr)
* level15->level16: `nc localhost 30000` (password found: BfMYroe26WYalil77FoDi9qh59eK5xNr) https://overthewire.org/wargames/bandit/bandit16.html
