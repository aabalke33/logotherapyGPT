css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex
}
.chat-message.user {
    background-color: #554763
}
.chat-message.bot {
    background-color: #736383;
    margin-bottom: 2.5rem;
}
.chat-message.source {
    background-color: #ac9eba;
    margin-bottom: 2.5rem;
}
.chat-message .message {
  width: 100%;
  padding: 0 1.5rem;
  color: #fff;
}
'''
source_template = '''
<div class="chat-message source">
    <div class="message">{{MSG}}</em></div>
</div>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</em></div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''
