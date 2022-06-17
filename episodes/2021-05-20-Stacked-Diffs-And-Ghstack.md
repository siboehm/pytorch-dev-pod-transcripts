---
layout: post
title: "Stacked Diffs And Ghstack"
date: 2021-05-20
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Stacked Diffs And Ghstack

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want to talk about something a little different, namely instead of talking about Pytorch itself, I wanna talk about one of the tools that we use to help develop Pytorch, namely GH Stack.
GH Stack solves a problem that goes something like this.
Imagine that you're working on some code in your project and, you know, you go tack a tack type type type and you've got a working implementation and you send it up for review as a pull request on GitHub.
And while you're waiting for people to come and actually review your beauty full code, You're like, okay.
Well, I'd kind of like to start working on the next feature, which is going to build on top of this patch that I did before.
Maybe this patch was some refactoring you know, a little bit of infrastructure that you needed for the next thing you're gonna work on.
So, okay, back to your local development copy, hack a hack work on your next you know, piece of the piece of the puzzle and maybe you're now done with that piece as well.
Now what do you do? Well, imagine that, you know, your first PR still hasn't been reviewed, or maybe it has been reviewed, but it still hasn't landed to Py Church.
because lands on PyTorch take a really long time.
Don't ask.
It just it takes a really long time.
So, you know, you've got the second poll request and sorry.
You got the second patch on top of your first patch.
It's logically independent, so like, you know, it can be reviewed in isolation from the first patch.
but you'd kinda like to put it out there and let people take a look at it.
How can you do this? Well, GitHub doesn't really make it easy for you to handle a situation like this because, you know, poll requests are, hey, here's a branch and then here's a master, I'm gonna compare against it, and then that's what the diff's gonna be.
So there's no easy, like, built in workflow for submitting extra patches on top of each other for review without, you know, necessarily forcing someone to review all the code from your previous patch as well.
So g h stack implements what we call stack diff development and it solves this problem.
Namely, by allowing you to create pull requests that are stacked on each other so that, you know, you can submit your first pull request.
then you can submit your next pull request, which depends on the first pull request.
And if someone wants to, you know, review, they can review each of these pull request separate but the second pull request still has all the changes from the first pull request so you can build on top of your work.
Staktiv development is not really an invention of the Pytruj project.
You know, a lot of people have used it before, in particular, if you use the code review tool called fabricator developed by Facebook originally.
That also implements the stack diff model.
And really, GitHub is a little behind the times and still not supporting stackives.
I heard that they've got some feature development in the works for supporting this workflow, but right now it doesn't work.
And so, you know, we have tools like g h stack to make this easier for us.
Okay.
So how do you use g h stack? Well, let's imagine that, you know, you're doing this story that I just told you before which is you hacked on some feature a and then you hacked on another feature b which depended on feature a.
So normally, you know, well, everyone uses get a little differently.
And so, like, one workflow that you might do is you say, okay.
Well, I'm just gonna just keep, you know, committing stuff like edit, okay, rework, you know, food, bar until, you know, you get to the end.
Right? And then you, like, do a bunch of extra commits on top.
and then you push all those commits to the pull request to be, you know, reviewed.
And then, you know, adding new updates isn't so hard.
Right? You just make a new change, you commit it, and then you push it to your pull request.
So g h stack requires you to work a little differently.
Instead of maintaining blow by blow, commit his three of every change you made.
Instead, g h stack wants you to create a single commit per per logical change that you want to submit.
a fabricator.
So let's say sorry.
Submit to GitHub.
So let's say that, you know, you've got three changes.
Right? Two, like, refactors that are independent of each other and then a feature implementation.
You'd structure these so that you have a commit one, which is refactor one, a commit two, which is refactor two, and then a commit three, that is, you know, the actual feature request in question.
And then once you have these three commits and they're all ready to go, you run g h stack.
Well, if you need to install it, you pip install g h stack.
and then you run g h stack.
And what g h stack will do is it'll look for every commit that, you know, is off of your branch from master, and it'll create a pillow request for each one.
So if you have three commits, it'll create three polling requests.
Then when you wanna make changes to the polling request later, just go ahead and amend or interactively rebase them.
By the way, about interactive rebasing.
Interactive rebasing my might sound, you know, tricky and complicated if you've never done this sort of thing and get before, but it's actually very easy to use.
And the way an interactive rebase works is you write git rebase dash i, and git will give you a list of all the commits that you've made onto the master And then all you need to do is say, okay.
Well, this is the commit that I wanna edit, and this is the commit that I don't care about.
And so, you know, you say edit, and then git rebase will drop you into a working tree with only, you know, the commit that you wanna edit, and so you can go ahead and edit it, amend the commit, and then continue your base further on.
So the way I tend to do interactive rebases is that I you know, first, I work on my patches.
Like, you know, okay, patch one.
I'm done.
commit it or come patch two, I'm done, committed, and so forth.
Then once I get to the end, usually what I do is if I notice that I need to fix up uncommit one and it's a small one, I'll just make a little edit at the very top.
I'll commit it so that I have a separate fix up commit, you know, standing on its own.
I don't run g h stack yet.
Instead, I, you know, run my build, make sure everything works.
And then I do an interactive rebase to move my fix of commit to the commit that it actually logically belongs to and then amended it using the so called fix up option in the rebased option.
And so this, you know, makes it easy for me to keep track of all the changes that I wanna do.
You have to make sure not to, like, overly merge conflict with yourself when you're doing this kind of thing, but it gets easier with practice.
So anyway, so that's it.
Right? So you've got these three commits and then get gives you some tools for modifying, you know, commits in the middle of the stack.
And I mostly try not to, like, make modifications and, you know, it's mostly aware of me letting letting myself get ahead of myself when I'm working on code.
By the way, in fabricator and mercurial land, there's actually support for actually going backwards and forwards in history using the h g preve and h g next command.
So this is actually a much better user interface than get.
Uh-huh.
Sorry.
Get well, Get's user interface is famously bad, so it's not surprising if I'm bad amounting it.
So, like, if you wanted to amend your commit, instead of amend a previous commit, one that wasn't at the top of your stack.
Instead of having to do the GitHub thing of setting them an interactive rebase or like making a fix up commit and then moving to the right place, All you have to do in Mercurial is say, h g preve, and then it'll put you in the previous commit.
You can go ahead and modify it.
And then mirrorial.
If you have enough extensions installed, we'll automatically restock all of your later commits on top of this one.
This is very convenient and I like it better.
If you want to try to, like, replicate this developer experiments and get, there are some quilt tools.
Apparently, I've never used any of them, but I think they're trying to do something similar.
So anyway, so you've got the stack of diffs.
Right? So you've got a stack of commits.
you want Jay Stack on them, they all get posted to GitHub, and that's it, edit them, and then Jay Stack again to, you know, put some more things on.
And for example, if you need to update to the latest version of master, you just need to use a non interactive rebase in this case.
So you know, you got your commits and you say get rebase, you know, master, origin master, if you're, you know, just get fetching like I normally do.
and then I'll just move all your commits over.
And of course, you might have to resolve some merge conflicts, but, you know, it's it's it's pretty straightforward and, you know, not much more difficult than merging.
One downside to rebasing in this way is you have to resolve merge conflicts for each commit individually.
On on like a merge commit where you just do everything all at once.
This makes sense because, you know, when we actually land a stack of disc from Jade Stack, we will land each commit separately.
So they will show up durably in the final GitHub history.
That's good for us because, you know, you went through all the trouble of making sure CI was passing on every commit.
So we will go through the trouble of making sure we preserve history in the situation.
Okay.
So that's basically how g h stack works.
You can get it once again by PIP installing j stack.
There is one caveat though, which is that in order for g h stack to work, you need push permissions to the Pytorch Pytorch repository.
So most people just, you know, fork Pytorch and push their stuff there.
And unfortunately, g h psych doesn't work because the way it works is that we create a bunch of branches representing what you're trying to merge into and then what your actual commits are.
Right? And the what you're gonna merge into branch has to actually live on Pytorch, Pytorch because if it doesn't, when you open a pull request, you open a pull request in your fork and not in Pytorch itself.
Okay.
Well, can I get right permissions? Well, if you're working on some feature that, you know, might be useful for stacking and you you know, have talked to someone on the Pietro's team about it, like, say, on an issue.
You know, you can just ask for write commits.
And we basically give write commits write access to anyone who asks for it because you can't actually write to master directly.
There's some complicated processed by which commits are sucked into our internal build system in EPI code and then spit it back out via this piece of software called Shipit.
So you can't touch the master branch You can just create temporary branches.
And so if you need to use g h stack to organize one of your PRs, just ask someone and we'll add you to the project.
Okay.
So what are some things to know about when using jstack? Well, one thing is that when you rebase your jstack or you make a modification to a commit very early to the g h stack.
We will push an update to every subsequent PR in your stack.
So please use this with care.
Right? Like each PR you push will trigger off a full CI run for everything in PyTorch Our CR runs are not that cheap.
So, you know, like, try to be nice and don't, you know, repeatedly, repush your stacks when, you know you know, oh, yeah, I just need a little bit of modification here.
Maybe defer that till later once you've finished all of your modifications and then push the g h stack all at once.
Another common thing to do is you've got your g h stack and you're just working on the latest commit.
As long as you don't rebase it onto master, you can safely g h stack in this case, and it'll only push the, you know, latest commit that you modified in that situation.
What are some other things to know about Jstag? Well, Jstag is also an open source project.
It lives on GitHub at easyang slash jstack.
Yeah.
I I sort of wrote this tool because I was so mad at having to deal with not being able to do stack diffs on GitHub.
So I I wrote it just to solve my own problem.
And it's, you know, it's not very much code and, you know, you can also check that out.
and use it on your own projects.
g h stack supports other repositories.
You just have to use a special command to land g h stack diffs.
because the normal merge to get hub button doesn't work.
For Pytorch cartridge, this doesn't matter because the normal merge to get hub but it doesn't work for completely unrelated reasons related to the ship at situation.
That's something you have to know about there.
Okay.
So g stack.
It lets you use your stack development.
Stack dev development is really good.
It lets you move ahead on what you're working on without blocking on COBOView.
your code actually landing to master.
It makes me a lot more productive and maybe it will make you a lot more productive if you're not just working on one off patches in Pytorch.
So give it a try.
That's all I wanted to say.
Talk to you next time.
.
