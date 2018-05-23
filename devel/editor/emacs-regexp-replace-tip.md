Title: Emacs 正则表达式搜索替换的小技巧
Date: 2018-05-23
Modified: 2018-05-23
Category: 效率
Tags: emacs

在Emacs中使用正则表达式替换时并不会像`isearch-forward-regexp`那么直观，无法查看到输入的正则表达式是否正确。之前我经常用`re-builder`进行表达式的测试，但是这样会打断当前的编辑工作。

经过验证发现可以使用`isearch-forward-regexp`代替`replace-regexp`，可以先用`isearch-forward-regexp`，它能交互式的验证所输入的表达式能否匹配，在匹配上第一个匹配位置时，输入`M-%`切换为`query-replace`模式，在输入要替换表达式后即可进行替换操作。
