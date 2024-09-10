"""_summary_

Returns:
    _type_: _description_
"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import autogen
from autogen import Agent, ConversableAgent

try:
    from termcolor import colored
except ImportError:

    def colored(x, *_, **__):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return x


class UserProxyWebAgent(autogen.UserProxyAgent):
    """_summary_

    Args:
        autogen (_type_): _description_
    """
    def __init__(self, *args, **kwargs):
        super(UserProxyWebAgent, self).__init__(*args, **kwargs)
        self.client_sent_queue = None
        self.client_receive_queue = None
        self._reply_func_list = []
        self.register_reply(trigger=[Agent, None],
                            reply_func=UserProxyWebAgent.a_check_termination_and_human_reply,
                            position=0)
        self.register_reply(trigger=[Agent, None],
                            reply_func=ConversableAgent.generate_oai_reply,
                            position=1)

    async def a_check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Check if the conversation should be terminated, and if human reply is provided."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        reply = ""
        no_human_input_msg = ""
        if self.human_input_mode == "ALWAYS":
            reply = await self.a_get_human_input(
                f"Provide feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: "
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # if the human input is empty, and the message is a termination message, then we will terminate the conversation
            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:
            if (
                self._consecutive_auto_reply_counter[sender]
                >= self._max_consecutive_auto_reply_dict[sender]
            ):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    terminate = self._is_termination_msg(message)
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply or "exit"

        if no_human_input_msg:
            print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # stop the conversation
        if reply == "next" or reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, reply

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] += 1
        if self.human_input_mode != "NEVER":
            print(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)

        return False, None

    def set_queues(self, client_sent_queue, client_receive_queue):
        """_summary_

        Args:
            client_sent_queue (_type_): _description_
            client_receive_queue (_type_): _description_
        """
        self.client_sent_queue = client_sent_queue
        self.client_receive_queue = client_receive_queue

    async def a_get_human_input(self, prompt: str) -> str | None:
        last_message = self.last_message()
        if last_message["content"]:
            # very hack way to retrieve the sender name but couldnt find a better way for now
            last_message["sender"] = [k for k in self.chat_messages.keys()][0].name

            await self.client_receive_queue.put(json.dumps(last_message))
            reply = await self.client_sent_queue.get()
            if reply and reply == "DO_FINISH":
                return "exit"
            return reply
        else:
            return None
