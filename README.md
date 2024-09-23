# Llamao
Llama.cpp with multi-agent support for one GPU.

# Why?
Because multi-agent support is the way to go. Even with a single human, the left hand does not know what the right hand is thinking, as the old saying goes. Messages are transferred in the human brain from the left and right hemispheres across the corpus callosum, the central divider in your brain. Neither side of the brain is exclusively responsible for any particular behavior; instead they both share the load, but there's always another thought coming because no matter how certain you are, your conscience is always ready right on the other side of your head. The left side may be more analytical; the right side may be more creative. It's really up to you, but what is clear is that two heads are better than none.

# Llamao.cpp how to use...

1) get gpt4all
2) just use that.
just kidding, ok, I'll try to explain.

# Prerequisites
1) You need QT Creator, along with all the rest of the stuff GPT4All uses, so go ahead and get that and set it up first.
2) You also need Electron to run Lore's Realm, if you're going to be interfacing with it that way. It's not strictly necessary, but you'd have to reconfigure GPT4All considerably to use our internal command structures.

# Internal Command Structures
There are several new ways to send queries to this llama, since it's doing a lot more than just passing messages to and from a neural net. These commands can be sent at any time in any part of your message, except for the * and & commands, which must be sent on their own and will not generate any response at all.

## /save
This command saves all actors and all their data. If you don't do this, they won't save automatically, however, some of the other commands will still work.

## /unload [actor]
Unloads a particular actor from the lists. The actor is saved first.

## /to [actor]\n
Sends the message, minus the "/to [actor]\n" parts, to just one actor instead of to all. If you are going to use this, make sure you use the proper formatting for your message: <|im_start|>yourname\nyour message, including /to target\n<|im_end|><|im_start|>target_actor\n ... if you don't the response may come from a strange angle or it might not work at all.

## Speaking of formatting.
The normal format for messages is now <|im_start|>yourname\nyour message<|im_end|>. There's no need to include an additional <|im_start|> for your bot's reply. Unless you are using /to as specified above, the LLM should handle choosing who speaks next.

## *key:actor_name\nvalue
stores a keyed memory to a particular actor.
note that if you send the same key with two different values, only the second value is kept.

## *global_key:\nvalue
stores a keyed memory for all actors.

## &actor_name\n
begins a sequence to load external memories from a text file. See main.js:/convert for details.
essentially it goes &actor_name\nfrom_name\nyear_month_day_hour_minute_second:message"<store_end>"\nfrom_name2....


# Contact
The versions may be misaligned by now which might make it impossible to setup. My version of GPT4All is from early 2024. I'm willing to help you set it up, just contact me at x.com/SendaoTrust

 Drop in the files from this repo's "llamao" into their appropriate places