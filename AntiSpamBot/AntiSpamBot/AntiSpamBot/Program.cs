using System;
using System.Linq;
using NumSharp;
using Tensorflow;
using TensorflowNET.Onnx;

class Program
{
    public static void Main(string[] args)
    {
        // Путь к модели .pb или .onnx
        var modelPath = "path_to_model.pb";  // Замените на ваш путь

        // Создаем сессию для загрузки модели
        var session = new InferenceSession(modelPath);

        // Пример входных данных (можно заменить на ваш обработанный текст)
        var paddedSequence = new int[] { 1, 2, 3, 4, 5 }; // Пример последовательности

        // Создаем numpy-массив с входными данными
        var inputData = np.array(paddedSequence.Select(x => (float)x).ToArray());

        // Создаем NamedOnnxValue для входа в модель
        var inputs = new NamedOnnxValue[]
        {
            NamedOnnxValue.CreateFromTensor("input_name", inputData)
        };

        // Запускаем модель и получаем предсказание
        var results = session.Run(inputs);

        // Получаем результат (например, вероятность того, что это спам)
        var output = results.FirstOrDefault();
        if (output != null)
        {
            // Преобразуем результат в удобный формат
            var resultArray = output.AsTensor<float>().ToArray();
            Console.WriteLine($"Предсказание: {resultArray[0]}"); // Печать вероятности или классификации
        }
    }

    // Пример простой токенизации текста (здесь можно использовать более сложный токенизатор, если нужно)
    static List<int> TokenizeText(string text)
    {
        // Преобразуем текст в список целых чисел (с использованием словаря токенов)
        // Это простая токенизация; в реальности вы должны использовать токенизатор, который вы использовали при обучении модели
        var wordIndex = new Dictionary<string, int>
        {
            { "<OOV>", 1 },  // Токен для неизвестных слов
            { "you", 2 },
            { "won", 3 },
            { "a", 4 },
            { "prize", 5 },
            { "congratulations", 6 }
        };

        var tokens = text.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var sequence = new List<int>();

        foreach (var word in tokens)
        {
            if (wordIndex.ContainsKey(word))
            {
                sequence.Add(wordIndex[word]);
            }
            else
            {
                sequence.Add(wordIndex["<OOV>"]); // Если слово неизвестно, используем токен для OOV
            }
        }

        // Паддинг для выравнивания последовательности
        return PadSequence(sequence, 10); // Например, максимальная длина последовательности = 10
    }

    // Паддинг для последовательности
    static List<int> PadSequence(List<int> sequence, int maxLen)
    {
        while (sequence.Count < maxLen)
            sequence.Insert(0, 0); // Паддинг в начале
        if (sequence.Count > maxLen)
            sequence = sequence.GetRange(0, maxLen); // Усечение длины
        return sequence;
    }
}





//using System;
//using System.IO;
//using System.Text.Json;
//using System.Collections.Generic;
//using System.Threading.Tasks;
//using Telegram.Bot;
//using Telegram.Bot.Exceptions;
//using Telegram.Bot.Types;
//using Telegram.Bot.Types.Enums;
//using Telegram.Bot.Types.ReplyMarkups;
//using static System.Net.Mime.MediaTypeNames;
//using System.Text;

//class Program
//{
//    private static readonly string BotToken = "7547438624:AAEUTUI2L2bDm1Fv6cG9uNXwCcdEwXvRTk0";
//    private static readonly long UserId = 5272975270;
//    private static readonly string JsonFilePath = "spam.json";

//    private static readonly TelegramBotClient Bot = new(BotToken);

//    static async Task Main(string[] args)
//    {
//        Console.WriteLine("Бот запущен...");

//        Bot.StartReceiving(
//            HandleUpdateAsync,
//            HandleErrorAsync
//        );

//        Console.ReadLine();
//    }

//    private static async Task HandleUpdateAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
//    {
//        // Проверяем, что это сообщение
//        if (update.Message is not { } message)
//            return;

//        // Проверяем, что это личное сообщение от указанного пользователя
//        if (message.Chat.Type != ChatType.Private || message.From.Id != UserId)
//            return;

//        // Получаем текст сообщения
//        var messageText = message.Text;
//        if (string.IsNullOrEmpty(messageText))
//            return;

//        // Формируем запись
//        var newEntry = new MessageEntry
//        {
//            Id = message.MessageId,
//            Text = messageText,
//            Label = "spam"
//        };

//        // Сохраняем сообщение в JSON
//        SaveMessageToJson(newEntry);

//        // Отправляем ответ пользователю
//        await botClient.SendTextMessageAsync(
//            chatId: message.Chat.Id,
//            text: "Сообщение сохранено и добавлено в JSON-файл.",
//            cancellationToken: cancellationToken
//        );
//    }

//    private static Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
//    {
//        var errorMessage = exception switch
//        {
//            ApiRequestException apiRequestException => $"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}",
//            _ => exception.ToString()
//        };

//        Console.WriteLine(errorMessage);
//        return Task.CompletedTask;
//    }

//    private static void SaveMessageToJson(MessageEntry messageEntry)
//    {
//        List<MessageEntry> messages;

//        // Загружаем существующие сообщения из файла
//        if (System.IO.File.Exists(JsonFilePath))
//        {
//            var json = System.IO.File.ReadAllText(JsonFilePath);
//            messages = JsonSerializer.Deserialize<List<MessageEntry>>(json) ?? new List<MessageEntry>();
//        }
//        else
//        {
//            messages = new List<MessageEntry>();
//        }

//        // Добавляем новое сообщение
//        messages.Add(messageEntry);

//        // Сохраняем в файл
//        var updatedJson = JsonSerializer.Serialize(messages, new JsonSerializerOptions { WriteIndented = true });
//        //System.IO.File.WriteAllText(JsonFilePath, updatedJson); 
//        System.IO.File.WriteAllText(JsonFilePath, updatedJson, Encoding.UTF8);

//    }
//}

//// Модель для хранения сообщения
//public class MessageEntry
//{
//    public int Id { get; set; }
//    public string Text { get; set; }
//    public string Label { get; set; }
//}

