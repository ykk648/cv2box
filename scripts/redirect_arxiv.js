// ==UserScript==
// @name        redirect_arxiv
// @namespace   redirect_arxiv
// @include     *
// @include     https://*github.io*
// @include     https://*arxiv.org/*
// @include     https://*google.c*
// @include     https://*semanticscholar.org/*
// @include     https://*github.com*
// @include     https://*zhihu.com*
// @include     https://*outlook.cn*
// @version     1.0
// @grant       none
// ==/UserScript==

// 重定向 arxiv.org 到 xxx.itp.ac.cn（中科院理论物理研究所镜像）


function findFatherNode(node, nodeName='A', maxDeep=1000){
    for (var i = 0; i < maxDeep; i++) {
        if (! node){return node}
        if (node.nodeName == nodeName){
            return node
        }else{
            node = node.parentElement
        }
    };
}

document.body.addEventListener('mousedown', function(e){
    var targ = e.target || e.srcElement;
    var aTag = findFatherNode(targ, 'A', 10);
    if (!aTag || !(aTag.href)){return};

    var headN = 17;
    var hrefHead = aTag.href.slice(0, headN);
    var hrefTail = aTag.href.slice(headN);
    if ( (hrefHead.indexOf('arxiv.org')==-1)){return};

    if ( hrefHead.match(/https?:\/\/arxiv\.org/) ) {
        hrefHead = hrefHead.replace(/https?:\/\/arxiv\.org/, 'http://xxx.itp.ac.cn');
    }
    aTag.href = hrefHead + hrefTail
    // console.log(targ, targ.href);
});