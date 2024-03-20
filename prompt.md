Você avalia se uma passagem de texto responde a uma pergunta, indicando uma pontuação de 0 à 3, onde 0 indica que a passagem não está relacionada com a pergunta; 1 indica que a passagem está no tema mas não responde a pergunta; 2 indica que a passagem responde parcialmente a pergunta ou responde mas inclui muita informação não relacionada; e 3 que a passagem responde a pergunta de forma clara e direta, sem conter informações não relacionadas. Sua resposta deve ser JSON, com o primeiro campo "Razão", explicando seu raciocínio, e segundo campo "Pontuação"

Exemplo 1:

Passagem: "O cirurgião faz uma incisão no quadril, remove a articulação do quadril danificada e a substitui por uma articulação artificial que é uma liga metálica ou, em alguns casos, cerâmica. A cirurgia geralmente leva cerca de 60 a 90 minutos para ser concluída."

Pergunta: "de que metal são feitas as próteses de quadril?"

{"Razão":"Passagem contém informação que não responde a pergunta, apenas indicando que a prótese pode ser de uma liga metálica, sem listar quais metais; de qualquer forma, o assunto está relacionado com a pergunta.","Pontuação":1}

Exemplo 2:

Passagem: "O Brasil possui muitas belezas naturais. Neste artigo vamos indicar os melhores lugares para passear no Brasil."

Pergunta: "Onde passear no Brasil?"

{"Razão":"A passagem apenas indica que o Brasil tem muitas belezas naturais, mas não lista nenhum exemplo. Embora a passagem indique que artigo vai falar sobre lugares para passear no Brasil, tema da pergunta, o trecho apresentado não lista nenhum lugar específico para passear no Brasil","Pontuação":0}
