---
layout: post
title: "Pytorch Probot"
date: 2021-07-11
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Pytorch Probot

Hello, everyone, and welcome to the Pytorch Probot.
Today, I want to talk about Pytorch Probot, a simple bot based on Probot that we use at Pytorch to do various operations on GitHub.
So what's the point of having a bot on GitHub that will do actions for you automatically? Well, as some members of the rest community have put very eloquently, a bot is a really good way of codifying otherwise very mechanical, very easy, but like, you know, time consuming tasks that humans would otherwise have to do into an easy to do framework that will do it automatically for you.
Right? So it's kind of like a lint rule, right, where when you have a linter in part of your CI, you don't have to then manually say, hey, you know, I think that line length is too long in your pull request.
The machine will automatically do that for you and, you know, you can save human bandwidth for things that actually matter.
Well, Probot is exactly this.
Its goal is to automate things that are otherwise easy to do and, you know, save our time for doing things like actually reading the issues you sent all of us.
There's a few pieces of functionality that we currently have implemented in Peter's Probot.
there's three.
So the first is what we call CC Bot.
So CC Bot is very simple.
When you have an issue on PyTorch, you can add labels to it.
Well, CC bot lets you maintain a subscription to any number of labels that you want to get CC done.
And then when someone labels an issue that way, well, CC bot will edit the issue and CCU on it.
So that's very useful because otherwise there isn't actually a good way to watch a label on GitHub.
and you don't actually want to be, you know, waiting through all of the issues on GitHub.
And if you are, you know, Even if you are pretty good about, like, looking over the issue list by hand, if you have a lot of labels you wanna keep abreast of, Well, that's a pretty complicated, you know, search query that you need for that situation.
So it's easier to just get them all in your inbox and then you can decide what you wanna do with them.
I subscribe to a lot of issues this way.
Really, it's there's too many issues in Pytuch for any one person to process So CCBot is a really good way of making sure you get CC ed on the stuff you're interested in even if you're not keeping an eye out for them in the ingestion point.
The second piece of print functionality that probate does is a label bot.
So what label bot does is if x is labeled with blah, then also label it with blah.
And one of the, like, use cases we do for this is for high priority.
So the way that high priority works in the Pinterest repository is once again, we have a lot of issues.
A lot of these issues are very minor and don't really matter that much.
And to make sure we don't lose the important issues and the big CF issues, we have a high priority label.
So what you do is when you think something is something that should actually get fixed, you can label it with high priority.
Now the problem is people don't necessarily always agree on, you know, what high priority is.
And we also have a socialization problem, which is like, you say you're new to the Pytruj project and, you know, you want to know whether or not something is high priority or not.
How the heck are you actually gonna know this? Right? Like, you're gonna be like, oh, well, I don't really know what it means to be high priority.
And then you might be conservative and you might not not mark an issue as high priority.
When actually it is high priority, And the problem is no one else is reading the issue because you were the one who was supposed to triage it, and then we just lose that issue to the sands of time.
So the idea behind the label bot is, well, whenever someone marks something as high priority, we also add a label triage review.
And what that means is that in our weekly triage meeting, we need to go over this issue and discuss why we think it's high priority and, you know you know, what what we're gonna do about it.
Actually, not so much what we're gonna do about it, but just, you know, why is it high priority? And the function of this, because most of the time when people label things high pretty.
They stay high pretty.
Like, I'd say ninety percent of issues are like that.
But the point of this is that everyone can easily see, hey, here are all the high priority issues that are going on.
This is what we collectively as a team think of as high priority.
It's a really good way of socializing issues in this way.
But we couldn't do this if, you know, when someone labels something as high priority, we didn't also say, please review it in the charge meeting.
And finally, there's a new feature that was developed by Eli Irrigas and Sam Estep for triggering CI jobs when a label is added.
So one of the things that we have as a problem is we want to build on a lot of configurations, but actually building all those configurations is pretty expensive.
So we don't want to actually build everything initially.
And if there is some exotic configuration that you think your PR actually needs testing on, well, you can add a label to your pull request, and that will trigger extra text.
And how those tests triggered? Well, that's done by Probot once again.
So that's really it.
Pro bots logic is not that complicated.
And I just wanna talk a little bit about the Probot framework, which I decided to use after much humming and hoeing.
You'll see why in a moment.
And also some sort of meta points about how Probot was designed and, you know, why I think these design ideas are actually good ones for the family.
So first, why probot? And actually, this was a not an easy choice for me because on Probot, the framework is a JavaScript framework.
And well, you know, we're PyTorch.
We're a Python shop.
So I would have ideally liked it if I could have ridden my bot in Python.
But Probot won me over by a number of pretty useful features.
that I, in fact, did appreciate a lot when I was developing this extension.
So for one, I can actually when I'm work developing the ProBOC framework, I can actually run my Node app locally.
So like, you know, I'm I'm hacking on my laptop.
I got my source code.
I made a change and I can run it And then, you know, I got my GitHub app going, and I can actually associate it with a real GitHub repository and do, you know, smoke testing by modifying the GitHub repository that triggers some hooks which get bounced to my local instance on my laptop processed whatever, you know, API calls I'm gonna do and then actually see it show up on GitHub.
And the way this is done is they have this, like, reflector service, which knows how which is, you know, like, if you install one of these dev instances, you register the reflector service as the host name because typically your MacBook isn't publicly addressable, and then it it bounces the request back to your, you know, local instance which is, you know, subscribed which which I subscribed to the reflector directly.
That's pretty awesome and it made developing very easy because normally when you're developing these hooks, it's very annoying to, like, generate synthetic events because, oh, you know, you gotta, like, go and trigger the hook and then download it from GitHub and then save it to some fixture blah blah blah.
Here you can just like directly just muck around with a repository and see what actually happens with Probot on the fly.
That's pretty nice.
Probot also has some opinions about testing, mostly, you know, based on mocking.
Marking isn't my favorite way of doing testing because it's very manual.
You have to, like, you know, create the fixtures, create what the outputs are.
and, you know, get that all going.
But it's a very convenient way when you're dealing with an external service like GitHub.
And, no, you don't actually want to be hitting the actual GitHub Endpoint API if you're actually, you know, running your test suite.
Of course, what we even better is if someone wrote a crappy reimplementation of GitHub with support for the GitHub API and the GitHub hooks and the GitHub.
notifications so that I could, like, just stand up a, like, a local copy of GitHub and then, you know, test against that.
Well, so I can always dream.
I actually have, like, a very small implementation of a very small fragment that is what I need for implementing GH Stack.
But, you know, that you you can hear about that in my GH Stack podcast.
Look, is in the past.
And finally, Probot, you know, had existing documentation for how to deploy it on AWS Lambda.
And this was very attractive to me because I was When I was building ProBaud, it was kinda one of these things where it's like, okay, I wanna build this thing and then I wanna forget about it and not have to worry about it ever again.
And if I had to, like, stand up a server and then actually, you know, maintain the server over time, well, I had to take software upgrades and, like, you know, kick the server when it goes down.
Don't wanna deal with that.
But if it's a Amazon Lambda, that's great.
And I don't have to worry about it.
Well, I mean, I do have to worry about it if Lambda, like, changes how their API works, but at least I don't have to worry about doing a server.
The so vented serverless, which, you know, A lot of people are like, actually, you know, we've gone too far.
You know, serverless is not so great.
But I think in this particular case, serverless was a really good call because it just reduces the maintenance cost.
And that really like gets me to the meta points.
Right? Like, one of the like enduring goals with the What the design of Project Probot was that I wanted to have as little maintenance as possible on this.
And so one one answer for that is to, you know, put this as a serverless deployment so that I don't have to worry about administering the server.
Another thing is that Probot has no state.
There's no database, there's no persistent state.
The CC bot is an interesting thing, which is like, you know, we need to know what we're gonna who we're gonna CC when a label is done.
And the way probate actually does this is we have this GitHub issue, and the GitHub issue's body contains the text of all the subscriptions.
And so what Probot just does is it loads up the GitHub issue on start up time, and then, like, that's that's how the state gets managed.
And this is very, very simple.
We can install a webhook for listening to issue updates so that when the issue gets updated, we know who we do it.
And then when the Lambda Instance dies, well, you know, the next time it spins up, we'll just refresh it again from GitHub.
No biggie.
So we've, like, offloaded the state onto GitHub, which, you know, is a big company and actually in the business of running a bunch of web servers and databases.
to maintain GitHub, and now we no longer have to maintain a database ourselves.
I can't stress how useful it is.
And of course, we give up some stuff to do this.
Right? Like, for example, you can't actually subscribe to labels on cc bots.
unless you actually have right permissions on the PyTorch repository because otherwise you can't edit this issue.
But, like, we we hand those out, like, candy.
So, like, if if you wanna, like, do one of those things, just ask.
Or you can just ask one of us to, like, add you to the CC list, and we can do that for you as well.
And another thing is that, you know, we don't want probought to be this thing where it can break in an unpredictable way and then, like, go into an infinite loop, like repeatedly adding labels and everything.
So ProBaud is designed to be item potent.
So if I accidentally deliver the webhook again or like I'm running multiple copies of Probot, which is what I was actually doing, at some period of time.
Probot can be deployed on Lambda.
It can also be deployed on GitHub actions.
I tried deploying it on GitHub actions.
And at the time, GitHub actions had a really long latency.
Like, it took, like, up to a minute before the GitHub action ran.
And, like, I really liked adding label and seeing it instantly show up like, you know, less than a second later.
So Lateral is the only way to do it.
But I I had both of these running at some time.
And if the bot wasn't item potent, then, you know, like, bad things could have in this case.
But, like, if it's item potent, that does make the types of operations you're allowed to do with the bot, you know, less complicated.
but it also just, you know, makes it harder to have accidents with a framework in question.
And finally, I talked a little bit about ProBots testing framework.
So I was wondering if I would just do live testing and then holiday.
And I was like, I mean, I know I actually want to test this code.
There is some non jubilant parsing code associated with like CC bot.
So I I went and, like, got the testing set up.
I, like, figured out how to do testing in Node JS, which, like, was kind of annoying because I'm not really a node person.
I don't really know about like NPM, but like it's all there and that was really nice.
because I doubt anyone else would have spent the time to add the testing framework.
So it's like making sure the initial infrastructure exists beforehand is really helpful when you want to hand the project off to someone else, and they're probably not actually gonna, like, add tests unless there's already a testing framework there.
So I think that paid off.
Probot can use developers.
For example, something that I've wanted to do for a really long time and haven't done because I've just never gotten around to it, is we have a CC bot and it works for issue labels, but it doesn't work for poor requests.
because I just never set up listenings for labels on pull requests.
So that would be really nice feature to have so that people could also tag pull requests and you could get cc'd on them.
in that case as well.
Alright.
That's everything I wanna talk about today.
Talk to you next time.
.
